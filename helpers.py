# helpers.py
import requests
import streamlit as st

API_V2 = "https://api.bexio.com/2.0"
API_V3 = "https://api.bexio.com/3.0"

def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

def fetch_company_profile(access_token: str) -> dict | None:
    """Try v2 first, then v3 fallback. Returns a dict (normalized)."""
    headers = auth_header(access_token)

    def _normalize(x):
        # bexio may return [ { ... } ]
        if isinstance(x, list) and x:
            return x[0] if isinstance(x[0], dict) else None
        return x if isinstance(x, dict) else None

    # v2
    try:
        r = requests.get(f"{API_V2}/company_profile", headers=headers, timeout=20)
        if r.status_code < 400:
            return _normalize(r.json())
    except Exception:
        pass

    # v3
    try:
        r = requests.get(f"{API_V3}/company_profile", headers=headers, timeout=20)
        if r.status_code < 400:
            return _normalize(r.json())
    except Exception:
        pass

    return None


def set_company_from_tokens(tokens: dict) -> None:
    """
    Stores Mandant info into:
      - st.session_state.company_profile
      - st.session_state.company_name
      - st.session_state.company_id

    No user/email stored.
    """
    access_token = (tokens or {}).get("access_token")
    if not access_token:
        return

    prof = fetch_company_profile(access_token)
    st.session_state.company_profile = prof or {}

    name = ""
    cid = ""

    if isinstance(prof, dict) and prof:
        # try common fields
        for k in ("name", "company_name", "company", "profile_name", "display_name"):
            v = prof.get(k)
            if v:
                name = str(v).strip()
                break

        # name parts used by some tenants
        if not name:
            n1 = str(prof.get("name_1") or "").strip()
            n2 = str(prof.get("name_2") or "").strip()
            name = (n1 + (" " + n2 if n2 else "")).strip()

        # nested structure fallback
        if not name and isinstance(prof.get("company"), dict):
            v = prof["company"].get("name") or prof["company"].get("company_name")
            if v:
                name = str(v).strip()

        # id variants
        for k in ("id", "company_id", "uuid"):
            v = prof.get(k)
            if v:
                cid = str(v).strip()
                break

    st.session_state.company_name = name
    st.session_state.company_id = cid
