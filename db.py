# db.py
import os
import json
from datetime import datetime, date
from sqlalchemy import create_engine, text

DEFAULT_DB_URL = os.getenv("DATABASE_URL", "sqlite:///imports.db")


def _json_default(o):
    """
    Convert numpy/pandas/date types to JSON-serializable Python types.
    Prevents: Object of type int64 is not JSON serializable
    """
    # numpy
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

    # pandas
    try:
        import pandas as pd
        if pd.isna(o):
            return None
        if isinstance(o, (pd.Timestamp,)):
            return o.to_pydatetime().isoformat()
    except Exception:
        pass

    # datetime/date
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, date):
        return o.isoformat()

    # last resort: stringify
    return str(o)


def get_engine(db_url: str | None = None):
    """
    Works with:
    - Streamlit secrets DATABASE_URL (cloud)
    - env var DATABASE_URL
    - fallback sqlite:///imports.db
    """
    url = db_url

    if not url:
        # Try Streamlit secrets first (if available)
        try:
            import streamlit as st
            url = (
                st.secrets.get("DATABASE_URL")
                or st.secrets.get("connections", {}).get("sql", {}).get("url")
            )
        except Exception:
            url = None

    url = url or DEFAULT_DB_URL

    # sqlite needs check_same_thread=False for Streamlit
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True)


def init_db(engine):
    """
    Creates one table that works on SQLite and Postgres.
    We store results_json as TEXT to keep it cross-db-simple.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS imports (
        id           TEXT PRIMARY KEY,
        tenant_id    TEXT NOT NULL,
        tenant_name  TEXT,
        created_at   TEXT NOT NULL,
        row_count    INTEGER NOT NULL DEFAULT 0,
        results_json TEXT,
        csv_bytes    BLOB
    );
    CREATE INDEX IF NOT EXISTS idx_imports_tenant_created
      ON imports (tenant_id, created_at DESC);
    """
    with engine.begin() as conn:
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            conn.execute(text(stmt))


def insert_import(engine, tenant_id: str, tenant_name: str, df_rows, results, csv_bytes: bytes):
    """
    Stores one import snapshot. Returns import_id (str).
    - results are JSON-dumped safely (handles np.int64 etc.)
    - csv_bytes stored as BLOB
    """
    import uuid

    import_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat(timespec="seconds")
    row_count = int(len(df_rows)) if df_rows is not None else 0

    results_json = json.dumps(results or [], ensure_ascii=False, default=_json_default)

    # ensure bytes for sqlite
    if isinstance(csv_bytes, memoryview):
        csv_bytes = csv_bytes.tobytes()

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO imports (id, tenant_id, tenant_name, created_at, row_count, results_json, csv_bytes)
                VALUES (:id, :tenant_id, :tenant_name, :created_at, :row_count, :results_json, :csv_bytes)
            """),
            dict(
                id=import_id,
                tenant_id=str(tenant_id),
                tenant_name=str(tenant_name or ""),
                created_at=created_at,
                row_count=row_count,
                results_json=results_json,
                csv_bytes=csv_bytes,
            ),
        )

    return import_id


def list_imports(engine, tenant_id: str, limit: int = 30):
    """
    Returns list of dicts: id, created_at, row_count, tenant_name
    """
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT id, tenant_name, created_at, row_count
                FROM imports
                WHERE tenant_id = :tenant_id
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            dict(tenant_id=str(tenant_id), limit=int(limit)),
        ).mappings().all()

    return [dict(r) for r in rows]


def get_import_csv(engine, import_id: str) -> tuple[bytes, str]:
    """
    Returns (csv_bytes, filename). Raises if not found.
    """
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, csv_bytes FROM imports WHERE id = :id"),
            dict(id=str(import_id)),
        ).mappings().first()

    if not row or row.get("csv_bytes") is None:
        raise ValueError("CSV not found for this import")

    csv_bytes = row["csv_bytes"]
    if isinstance(csv_bytes, memoryview):
        csv_bytes = csv_bytes.tobytes()

    fname = f"import_{row['id']}.csv"
    return csv_bytes, fname


def delete_import(engine, tenant_id: str, import_id: str) -> int:
    """
    Deletes one import (scoped by tenant_id).
    Returns number of deleted rows (0/1).
    """
    with engine.begin() as conn:
        res = conn.execute(
            text("""
                DELETE FROM imports
                WHERE id = :id AND tenant_id = :tenant_id
            """),
            dict(id=str(import_id), tenant_id=str(tenant_id)),
        )
        return int(res.rowcount or 0)



