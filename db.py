# db.py
import os
import json
from datetime import datetime, date, timezone
from sqlalchemy import create_engine, text, inspect

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
        try:
            import streamlit as st
            url = (
                st.secrets.get("DATABASE_URL")
                or st.secrets.get("connections", {}).get("sql", {}).get("url")
            )
        except Exception:
            url = None

    url = url or DEFAULT_DB_URL

    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True)


def init_db(engine):
    """
    IMPORTANT:
    Your Postgres DB already has a table with NOT NULL payload_json (jsonb).
    So we must keep payload_json in the schema and always write it.

    This init is cross-db:
    - Postgres: uuid + timestamptz + jsonb + bytea
    - SQLite: TEXT + TEXT + TEXT + BLOB
    """
    dialect = engine.dialect.name.lower()

    if "postgres" in dialect:
        ddl = """
        CREATE TABLE IF NOT EXISTS imports (
          id uuid PRIMARY KEY,
          created_at timestamptz NOT NULL DEFAULT now(),
          tenant_id text NOT NULL,
          tenant_name text,
          row_count int NOT NULL DEFAULT 0,
          payload_json jsonb NOT NULL,
          results_json jsonb,
          csv_bytes bytea
        );
        CREATE INDEX IF NOT EXISTS idx_imports_tenant_created
          ON imports (tenant_id, created_at DESC);
        """
        with engine.begin() as conn:
            for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
                conn.execute(text(stmt))
        return

    # SQLite (or other): keep same logical columns, simpler types.
    ddl = """
    CREATE TABLE IF NOT EXISTS imports (
        id           TEXT PRIMARY KEY,
        created_at   TEXT NOT NULL,
        tenant_id    TEXT NOT NULL,
        tenant_name  TEXT,
        row_count    INTEGER NOT NULL DEFAULT 0,
        payload_json TEXT NOT NULL DEFAULT '[]',
        results_json TEXT,
        csv_bytes    BLOB
    );
    CREATE INDEX IF NOT EXISTS idx_imports_tenant_created
      ON imports (tenant_id, created_at DESC);
    """
    with engine.begin() as conn:
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            conn.execute(text(stmt))

    # If an old SQLite file exists without payload_json, add it safely.
    try:
        insp = inspect(engine)
        cols = {c["name"] for c in insp.get_columns("imports")}
        if "payload_json" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE imports ADD COLUMN payload_json TEXT NOT NULL DEFAULT '[]'"))
    except Exception:
        pass


def insert_import(engine, tenant_id: str, tenant_name: str, df_rows, results=None, csv_bytes: bytes | None = None):
    """
    Stores one import snapshot. Returns import_id (str).

    - ALWAYS writes payload_json (because Postgres schema requires it NOT NULL)
    - results_json can be NULL
    - csv_bytes optional
    """
    import uuid

    dialect = engine.dialect.name.lower()

    # payload (what user posted / grid snapshot)
    try:
        import pandas as pd
        if isinstance(df_rows, pd.DataFrame):
            payload = df_rows.to_dict(orient="records")
            row_count = int(len(df_rows))
        else:
            payload = df_rows if df_rows is not None else []
            row_count = int(len(payload or []))
    except Exception:
        payload = df_rows if df_rows is not None else []
        row_count = int(len(payload or []))

    payload_json = json.dumps(payload or [], ensure_ascii=False, default=_json_default)
    results_json = json.dumps(results or [], ensure_ascii=False, default=_json_default) if results is not None else None

    if isinstance(csv_bytes, memoryview):
        csv_bytes = csv_bytes.tobytes()

    # ids / time
    imp_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()

    if "postgres" in dialect:
        # insert with casts to jsonb to satisfy existing schema
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO imports (
                        id, created_at, tenant_id, tenant_name, row_count,
                        payload_json, results_json, csv_bytes
                    )
                    VALUES (
                        :id::uuid, :created_at::timestamptz, :tenant_id, :tenant_name, :row_count,
                        CAST(:payload_json AS jsonb),
                        CAST(:results_json AS jsonb),
                        :csv_bytes
                    )
                """),
                {
                    "id": imp_id,
                    "created_at": now_iso,
                    "tenant_id": str(tenant_id),
                    "tenant_name": str(tenant_name or ""),
                    "row_count": int(row_count),
                    "payload_json": payload_json,
                    "results_json": results_json if results_json is not None else "[]",
                    "csv_bytes": csv_bytes,
                },
            )
        return imp_id

    # SQLite / other: plain TEXT columns
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO imports (
                    id, created_at, tenant_id, tenant_name, row_count,
                    payload_json, results_json, csv_bytes
                )
                VALUES (
                    :id, :created_at, :tenant_id, :tenant_name, :row_count,
                    :payload_json, :results_json, :csv_bytes
                )
            """),
            {
                "id": imp_id,
                "created_at": now_iso,
                "tenant_id": str(tenant_id),
                "tenant_name": str(tenant_name or ""),
                "row_count": int(row_count),
                "payload_json": payload_json,
                "results_json": results_json,
                "csv_bytes": csv_bytes,
            },
        )

    return imp_id


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
            {"tenant_id": str(tenant_id), "limit": int(limit)},
        ).mappings().all()

    return [dict(r) for r in rows]


def get_import_csv(engine, import_id: str) -> tuple[bytes, str]:
    """
    Returns (csv_bytes, filename). Raises if not found.
    """
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, csv_bytes FROM imports WHERE id = :id"),
            {"id": str(import_id)},
        ).mappings().first()

    if not row or row.get("csv_bytes") is None:
        raise ValueError("CSV not found for this import")

    csv_bytes = row["csv_bytes"]
    if isinstance(csv_bytes, memoryview):
        csv_bytes = csv_bytes.tobytes()

    return csv_bytes, f"import_{row['id']}.csv"


def delete_import(engine, tenant_id: str, import_id: str) -> int:
    """
    Deletes one import (scoped by tenant_id). Returns number of deleted rows (0/1).
    """
    with engine.begin() as conn:
        res = conn.execute(
            text("DELETE FROM imports WHERE id = :id AND tenant_id = :tenant_id"),
            {"id": str(import_id), "tenant_id": str(tenant_id)},
        )
        return int(res.rowcount or 0)


