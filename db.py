# db.py
import os
from datetime import datetime, date
from typing import Optional, Iterable

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

def get_engine() -> Engine:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL missing. Set it in Streamlit Secrets.")
    # Allow both postgres urls and sqlalchemy urls
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return create_engine(url, pool_pre_ping=True)

DDL = """
-- paste the SQL schema from above here, exactly
"""

def init_db(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(DDL))

def get_or_create_tenant(engine: Engine, company_id: str, company_name: str) -> int:
    with engine.begin() as conn:
        row = conn.execute(
            text("select id from tenants where company_id=:cid"),
            {"cid": company_id},
        ).fetchone()
        if row:
            # optionally keep name updated
            conn.execute(
                text("update tenants set company_name=:n where id=:id"),
                {"n": company_name, "id": row[0]},
            )
            return int(row[0])

        res = conn.execute(
            text("insert into tenants(company_id, company_name) values (:cid,:n) returning id"),
            {"cid": company_id, "n": company_name},
        ).fetchone()
        return int(res[0])

def get_or_create_user(engine: Engine, tenant_id: int, user_sub: str) -> int:
    with engine.begin() as conn:
        row = conn.execute(
            text("select id from users where tenant_id=:t and user_sub=:s"),
            {"t": tenant_id, "s": user_sub},
        ).fetchone()
        if row:
            return int(row[0])
        res = conn.execute(
            text("insert into users(tenant_id, user_sub) values (:t,:s) returning id"),
            {"t": tenant_id, "s": user_sub},
        ).fetchone()
        return int(res[0])

def create_import(engine: Engine, tenant_id: int, user_id: Optional[int], source: str, filename: Optional[str]) -> int:
    with engine.begin() as conn:
        res = conn.execute(
            text("""
                insert into imports(tenant_id, user_id, source, filename, status)
                values (:t,:u,:s,:f,'draft')
                returning id
            """),
            {"t": tenant_id, "u": user_id, "s": source, "f": filename},
        ).fetchone()
        return int(res[0])

def upsert_import_rows(engine: Engine, import_id: int, df: pd.DataFrame) -> None:
    """
    df should contain columns:
    csv_row, datum, beschreibung, betrag, currency, exchange_rate, soll, haben, mwst_code, mwst_konto
    """
    if df is None or df.empty:
        return

    rows = []
    for _, r in df.iterrows():
        row_no = int(r.get("csv_row") or 0) or 0
        if row_no <= 0:
            continue

        # parse date (YYYY-MM-DD already in your flow)
        d = r.get("datum") or None
        if isinstance(d, str) and d.strip():
            try:
                d = date.fromisoformat(d.strip())
            except Exception:
                d = None

        rows.append({
            "import_id": import_id,
            "row_no": row_no,
            "date": d,
            "description": (r.get("beschreibung") or "")[:4000],
            "amount": float(r.get("betrag") or 0),
            "currency": (r.get("currency") or "").upper(),
            "exchange_rate": float(r.get("exchange_rate") or 1.0),
            "soll": str(r.get("soll") or ""),
            "haben": str(r.get("haben") or ""),
            "mwst_code": str(r.get("mwst_code") or ""),
            "mwst_konto": str(r.get("mwst_konto") or ""),
        })

    if not rows:
        return

    with engine.begin() as conn:
        conn.execute(
            text("""
                insert into import_rows
                (import_id, row_no, date, description, amount, currency, exchange_rate, soll, haben, mwst_code, mwst_konto, post_status)
                values
                (:import_id, :row_no, :date, :description, :amount, :currency, :exchange_rate, :soll, :haben, :mwst_code, :mwst_konto, 'pending')
                on conflict (import_id, row_no) do update set
                    date=excluded.date,
                    description=excluded.description,
                    amount=excluded.amount,
                    currency=excluded.currency,
                    exchange_rate=excluded.exchange_rate,
                    soll=excluded.soll,
                    haben=excluded.haben,
                    mwst_code=excluded.mwst_code,
                    mwst_konto=excluded.mwst_konto
            """),
            rows,
        )

def mark_row_result(engine: Engine, import_id: int, row_no: int, status: str, bexio_id: Optional[str]=None, error: Optional[str]=None) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                update import_rows
                set post_status=:st,
                    bexio_manual_entry_id=:bid,
                    error_message=:err,
                    posted_at=case when :st='ok' then now() else posted_at end
                where import_id=:iid and row_no=:rno
            """),
            {"st": status, "bid": bexio_id, "err": error, "iid": import_id, "rno": row_no},
        )

def set_import_status(engine: Engine, import_id: int) -> None:
    """
    Derive status from row statuses.
    """
    with engine.begin() as conn:
        agg = conn.execute(text("""
            select
              sum(case when post_status='ok' then 1 else 0 end) as ok,
              sum(case when post_status='error' then 1 else 0 end) as err,
              sum(case when post_status='pending' then 1 else 0 end) as pend
            from import_rows
            where import_id=:iid
        """), {"iid": import_id}).fetchone()

        ok, err, pend = (agg[0] or 0), (agg[1] or 0), (agg[2] or 0)

        if pend == 0 and err == 0 and ok > 0:
            status = "posted"
        elif ok > 0 and (err > 0 or pend > 0):
            status = "partial"
        elif err > 0 and ok == 0:
            status = "failed"
        else:
            status = "draft"

        conn.execute(text("""
            update imports set status=:s, updated_at=now() where id=:iid
        """), {"s": status, "iid": import_id})
