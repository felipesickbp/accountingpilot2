# helpers.py
import base64, json
import streamlit as st

def jwt_claims(token: str) -> dict:
    """
    Decode JWT payload WITHOUT verifying signature (OK for UI display only).
    Returns {} if not a JWT or decode fails.
    """
    if not token or token.count(".") < 2:
        return {}
    try:
        payload_b64 = token.split(".", 2)[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)  # padding
        payload = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return {}

def extract_selected_company(tokens: dict) -> tuple[str, str]:
    """
    Best-effort: pull selected Mandant from id_token/access_token claims.
    Returns (company_name, company_id) as strings (may be empty).
    """
    claims = jwt_claims(tokens.get("id_token", "")) or {}
    if not claims:
        claims = jwt_claims(tokens.get("access_token", "")) or {}

    if not claims:
        return "", ""

    name = ""
    cid = ""

    # direct keys
    for k in ("company_name", "company", "tenant_name", "organization", "org_name", "display_name"):
        v = claims.get(k)
        if isinstance(v, str) and v.strip():
            name = v.strip()
            break

    for k in ("company_id", "tenant_id", "organization_id", "org_id", "bexio_company_id"):
        v = claims.get(k)
        if v is not None and str(v).strip():
            cid = str(v).strip()
            break

    # nested structures
    if not name:
        for k in ("tenant", "company", "organization"):
            v = claims.get(k)
            if isinstance(v, dict):
                n = v.get("name") or v.get("company_name") or v.get("display_name")
                if n:
                    name = str(n).strip()
                i = v.get("id") or v.get("company_id") or v.get("tenant_id")
                if i and not cid:
                    cid = str(i).strip()
                if name:
                    break

    return name, cid

def set_company_from_tokens(tokens: dict) -> None:
    """
    Writes st.session_state.company_name / company_id (only if found).
    """
    name, cid = extract_selected_company(tokens)
    if name:
        st.session_state.company_name = name
    if cid:
        st.session_state.company_id = cid
