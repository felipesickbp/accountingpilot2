# db.py
import os, uuid, json
from datetime import datetime, timezone, date
from sqlalchemy import create_engine, text

def get_engine():
    try:
        import streamlit as st
        url = st.secrets.get("DATABASE_URL") or st.secrets.get("connections", {}).get("sql", {}).get("url")
    except Exception:
        url = None

    url = url or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("Missing DATABASE_URL in secrets or env")

    return create_engine(url, pool_pre_ping=True)

def init_db(engine):
    ddl = """
    create table if not exists imports (
      id uuid primary key,
      created_at timestamptz not null default now(),
      tenant_id text not null,
      tenant_name text,
      row_count int not null default 0,
      payload_json jsonb not null,
      results_json jsonb,
      csv_bytes bytea
    );
    create index if not exists idx_imports_tenant_created
      on imports (tenant_id, created_at desc);
    """
    with engine.begin() as conn:
        for stmt in ddl.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))

def _json_default(o):
    """
    Convert numpy/pandas/date types to JSON-serializable Python types.
    """
    # Avoid importing numpy/pandas globally
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
    except Exception:
        pass

    try:
        import pandas as pd
        if pd.isna(o):
            return None
        if isinstance(o, (pd.Timestamp,)):
            # ISO string
            return o.to_pydatetime().isoformat()
    except Exception:
        pass

    if isinstance(o, (datetime,)):
        return o.isoformat()
    if isinstance(o, (date,)):
        return o.isoformat()

    # last-resort: stringify (keeps DB write from failing)
    return str(o)

def insert_import(engine, tenant_id: str, tenant_name: str, df_rows, results=None, csv_bytes: bytes | None = None):
    import pandas as pd

    if isinstance(df_rows, pd.DataFrame):
        payload = df_rows.to_dict(orient="records")
        row_count = int(len(df_rows))
    else:
        payload = df_rows
        row_count = int(len(payload or []))

    imp_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    payload_json = json.dumps(payload, default=_json_default)
    results_json = json.dumps(results, default=_json_default) if results is not None else None

    with engine.begin() as conn:
        conn.execute(
            text("""
                insert into imports (id, created_at, tenant_id, tenant_name, row_count, payload_json, results_json, csv_bytes)
                values (:id, :created_at, :tenant_id, :tenant_name, :row_count, :payload_json::jsonb, :results_json::jsonb, :csv_bytes)
            """),
            {
                "id": str(imp_id),
                "created_at": now.isoformat(),
                "tenant_id": tenant_id,
                "tenant_name": tenant_name,
                "row_count": row_count,
                "payload_json": payload_json,
                "results_json": results_json,
                "csv_bytes": csv_bytes,
            }
        )
    return str(imp_id)

def list_imports(engine, tenant_id: str, limit: int = 50):
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                select id, created_at, tenant_name, row_count
                from imports
                where tenant_id = :tenant_id
                order by created_at desc
                limit :limit
            """),
            {"tenant_id": tenant_id, "limit": limit}
        ).mappings().all()
    return rows

def get_import_csv(engine, import_id: str) -> bytes | None:
    with engine.begin() as conn:
        row = conn.execute(
            text("select csv_bytes from imports where id = :id"),
            {"id": import_id}
        ).mappings().first()
    if not row:
        return None
    return row["csv_bytes"]


