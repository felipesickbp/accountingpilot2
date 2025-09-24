import os, time, base64
import streamlit as st
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv
from datetime import date as dt_date
import pandas as pd

load_dotenv(override=True)

# =========================
# ENV + OAUTH HELPERS
# =========================

def _getenv(name: str, required=True, default=None):
    v = os.getenv(name, default)
    if required and (v is None or str(v).strip() == ""):
        st.error(f"Missing required env: {name}")
        st.stop()
    return v

BEXIO_CLIENT_ID     = _getenv("BEXIO_CLIENT_ID")
BEXIO_CLIENT_SECRET = _getenv("BEXIO_CLIENT_SECRET")
BEXIO_REDIRECT_URI  = _getenv("BEXIO_REDIRECT_URI")

AUTH_URL  = "https://auth.bexio.com/realms/bexio/protocol/openid-connect/auth"
TOKEN_URL = "https://auth.bexio.com/realms/bexio/protocol/openid-connect/token"

API_V3 = "https://api.bexio.com/3.0"
MANUAL_ENTRIES_V3 = f"{API_V3}/accounting/manual_entries"
NEXT_REF_V3       = f"{API_V3}/accounting/manual_entries/next_ref_nr"
ACCOUNTS_V3       = f"{API_V3}/accounting/accounts"


# v2 (read-only import of accounts)
API_V2 = "https://api.bexio.com/2.0"

def _auth_v2():
    return {
        **auth_header(st.session_state.oauth["access_token"]),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

def fetch_all_accounts_v2(limit=2000):
    """
    /2.0/accounts/search with 'not_null' to fetch all accounts; paginated by offset.
    """
    url = f"{API_V2}/accounts/search"
    offset = 0
    rows = []
    payload = [ {"field": "name", "value": "", "criteria": "not_null"} ]
    while True:
        params = {"limit": limit, "offset": offset}
        r = requests.post(url, headers=_auth_v2(), json=payload, params=params, timeout=30)
        r.raise_for_status()
        chunk = r.json() if isinstance(r.json(), list) else []
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < limit:
            break
        offset += limit
    return rows


SCOPES = "openid profile email offline_access company_profile"

st.set_page_config(page_title="bexio Bulk Manual Entries (v3)", page_icon="📘")

if "oauth" not in st.session_state:
    st.session_state.oauth = {}
if "bulk_df" not in st.session_state:
    st.session_state.bulk_df = None

if "acct_map_by_number" not in st.session_state:
    st.session_state.acct_map_by_number = {}
if "acct_df" not in st.session_state:
    st.session_state.acct_df = None

def auth_header(token):
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def save_tokens(tokens):
    tokens["expires_at"] = time.time() + int(tokens.get("expires_in", 3600)) - 30
    st.session_state.oauth = tokens


def need_login():
    return (not st.session_state.oauth) or (time.time() > st.session_state.oauth.get("expires_at", 0))


def refresh_access_token():
    if not st.session_state.oauth.get("refresh_token"):
        return
    data = {
        "grant_type": "refresh_token",
        "refresh_token": st.session_state.oauth["refresh_token"],
        "client_id": BEXIO_CLIENT_ID,
        "client_secret": BEXIO_CLIENT_SECRET,
        "redirect_uri": BEXIO_REDIRECT_URI,
    }
    r = requests.post(TOKEN_URL, data=data, timeout=30)
    r.raise_for_status()
    save_tokens(r.json())


def login_link():
    state = "anti-csrf-" + base64.urlsafe_b64encode(os.urandom(12)).decode("utf-8")
    params = {
        "client_id": BEXIO_CLIENT_ID,
        "redirect_uri": BEXIO_REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
    }
    url = f"{AUTH_URL}?{urlencode(params)}"
    st.markdown(f"[Sign in with bexio]({url})")


def handle_callback():
    code = st.query_params.get("code")
    if not code:
        return
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": BEXIO_REDIRECT_URI,
        "client_id": BEXIO_CLIENT_ID,
        "client_secret": BEXIO_CLIENT_SECRET,
    }
    r = requests.post(TOKEN_URL, data=data, timeout=30)
    r.raise_for_status()
    save_tokens(r.json())
    st.query_params.clear()


def _auth():
    return {**auth_header(st.session_state.oauth["access_token"]), "Accept": "application/json"}


# =========================
# PAGE
# =========================

st.title("📘 bexio Bulk Manual Entries (API v3)")

# OAuth
handle_callback()
if need_login():
    st.info("Verbinde dein bexio Konto, um Buchungen zu posten.")
    login_link()
    st.stop()
if time.time() > st.session_state.oauth.get("expires_at", 0):
    with st.spinner("Session wird erneuert …"):
        refresh_access_token()

# =========================
# Accounts table (all accounts & IDs)
# =========================
with st.expander("📒 Konto-IDs aus deinem Mandanten (v3-Quellen)"):
    try:
        rows = []
        # 1) Banking-linked GL accounts (reliable v3 endpoint)
        try:
            r = requests.get(f"{API_V3}/banking/accounts", headers=_auth(), timeout=30)
            r.raise_for_status()
            banks = r.json() if isinstance(r.json(), list) else []
            for b in banks:
                rows.append({
                    "source": "banking",
                    "account_id": b.get("account_id"),
                    "label": b.get("name") or b.get("bank_name"),
                    "note": f"IBAN {b.get('iban_nr')}"
                })
        except Exception as e:
            st.warning(f"Bankkonten konnten nicht geladen werden: {e}")

        # 2) Tax accounts (carry account ids)
        try:
            r = requests.get(f"{API_V3}/taxes?scope=active", headers=_auth(), timeout=30)
            r.raise_for_status()
            taxes = r.json() if isinstance(r.json(), list) else []
            for t in taxes:
                acc = t.get("account_id")
                if acc:
                    rows.append({
                        "source": "tax",
                        "account_id": acc,
                        "label": t.get("display_name") or t.get("code"),
                        "note": f"tax_id {t.get('id')}"
                    })
        except Exception as e:
            st.info("Steuerkonten konnten nicht geladen werden (optional): " + str(e))

        # 3) Journal scan (wide window)
        try:
            from datetime import date, timedelta
            start = (date.today() - timedelta(days=3650)).isoformat()  # ~10 Jahre
            r = requests.get(f"{API_V3}/accounting/journal?from={start}&limit=2000", headers=_auth(), timeout=30)
            r.raise_for_status()
            entries = r.json() if isinstance(r.json(), list) else []
            seen = set()
            for j in entries:
                for k in ("debit_account_id", "credit_account_id"):
                    a = j.get(k)
                    if a and a not in seen:
                        seen.add(a)
                        rows.append({
                            "source": "journal",
                            "account_id": a,
                            "label": j.get("description"),
                            "note": "sichtbar in Journal"
                        })
        except Exception as e:
            st.info("Journal konnte nicht geladen werden: " + str(e))

        if rows:
            df_ids = pd.DataFrame(rows).drop_duplicates(subset=["account_id"]).sort_values("account_id")
            st.dataframe(df_ids, use_container_width=True, hide_index=True)
            st.caption("Nutze die **account_id** in den Spalten ‘soll’/‘haben’. (Nur v3-Quellen.)")
        else:
            st.info("Keine Konto-IDs aus v3-Quellen gefunden.")
    except Exception as e:
        st.error(f"Fehler beim Ermitteln der Konto-IDs: {e}")


# ---- v2 Kontenplan import (read-only, builds number -> id mapping) ----
st.subheader("Kontenplan aus bexio importieren (READ via v2)")
st.caption("Nur zum Einlesen der Mapping-Tabelle (account_no → id). Postings bleiben über v3.")

col_i1, col_i2 = st.columns([1,1])
do_import = col_i1.button("Kontenplan aus bexio importieren")
clear_map = col_i2.button("Mapping leeren")

if clear_map:
    st.session_state.acct_map_by_number = {}
    st.session_state.acct_df = None
    st.success("Lokales Mapping zurückgesetzt.")

if do_import:
    try:
        with st.spinner("Lade Kontenplan …"):
            accs = fetch_all_accounts_v2()
        if not accs:
            st.warning("Keine Konten gefunden.")
        else:
            df = pd.DataFrame([{
                "id": a.get("id"),
                "number": a.get("account_no"),
                "name": a.get("name"),
                "type": a.get("account_type"),
                "active": a.get("is_active"),
            } for a in accs]).sort_values(["number", "id"], na_position="last")
            mp = {}
            for _, r in df.iterrows():
                n = str(r["number"]).strip() if pd.notna(r["number"]) else None
                i = int(r["id"]) if pd.notna(r["id"]) else None
                if n and i:
                    mp[n] = i
            st.session_state.acct_map_by_number = mp
            st.session_state.acct_df = df
            st.success(f"{len(df)} Konten importiert.")
    except requests.HTTPError as e:
        st.error(f"HTTP error: {e.response.status_code} – {e.response.text}")
    except Exception as e:
        st.error(f"Import fehlgeschlagen: {e}")

if st.session_state.acct_df is not None:
    st.dataframe(st.session_state.acct_df, use_container_width=True, hide_index=True)
    st.caption("Diese Tabelle liefert die **externe ID** (= `id`) pro Kontonummer.")

# Render editable grid
edited_df = st.data_editor(
    st.session_state.bulk_df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "buchungsnummer": st.column_config.TextColumn("buchungsnummer", help="Optional. Wenn leer, kann automatisch bezogen werden."),
        "datum": st.column_config.DateColumn("datum", format="YYYY-MM-DD"),
        "betrag": st.column_config.NumberColumn("betrag", min_value=0.0, step=0.05, format="%.2f"),
        "soll": st.column_config.NumberColumn("soll (debit_account_id)", min_value=1, step=1),
        "haben": st.column_config.NumberColumn("haben (credit_account_id)", min_value=1, step=1),
    },
)

st.session_state.bulk_df = edited_df

colA, colB = st.columns(2)
auto_ref = colA.checkbox("Referenznummer automatisch beziehen (wenn leer)", value=True)
post_btn = colB.button("Buchungen posten", type="primary")

def resolve_account_id_from_number_or_id(val):
    """
    Accept a raw account_id (int) OR an account number (string),
    return the account_id (int) using the imported mapping if present.
    """
    if val is None or str(val).strip() == "":
        return None
    # Already an int-like id?
    try:
        return int(val)
    except Exception:
        pass
    # Lookup by number (string)
    key = str(val).strip()
    return int(st.session_state.acct_map_by_number[key]) if key in st.session_state.acct_map_by_number else None


# =========================
# Posting logic (one API call per Zeile)
# =========================

if post_btn:
    rows = edited_df.fillna("")
    results = []

    for idx, row in rows.iterrows():
        try:
            # Skip completely empty lines
            if (str(row.get("datum", "")).strip() == "" and
                float(row.get("betrag", 0) or 0) == 0 and
                str(row.get("soll", "")).strip() == "" and
                str(row.get("haben", "")).strip() == ""):
                continue

            # Validate required fields
            d = row.get("datum")
            if isinstance(d, dt_date):
                date_iso = d.isoformat()
            else:
                date_iso = str(d)
                # Basic YYYY-MM-DD check
                if not isinstance(date_iso, str) or len(date_iso.split("-")) != 3:
                    raise ValueError("Ungültiges Datum – erwartet YYYY-MM-DD.")

            amount = float(row.get("betrag") or 0)
            if amount <= 0:
                raise ValueError("Betrag muss > 0 sein.")

            debit_id  = resolve_account_id_from_number_or_id(row.get("soll"))
            credit_id = resolve_account_id_from_number_or_id(row.get("haben"))
            if not debit_id or not credit_id:
                raise ValueError("Ungültiges Konto: konnte keine account_id für soll/haben auflösen (bitte Kontenplan importieren).")


            ref_nr = str(row.get("buchungsnummer") or "").strip()
            if auto_ref and not ref_nr:
                rr = requests.get(NEXT_REF_V3, headers=_auth(), timeout=15)
                rr.raise_for_status()
                ref_nr = (rr.json() or {}).get("next_ref_nr") or ""

            payload = {
                "type": "manual_single_entry",
                "date": date_iso,
                "entries": [
                    {
                        "debit_account_id": debit_id,
                        "credit_account_id": credit_id,
                        "amount": amount,
                        "description": "",  # bewusst leer; bei Bedarf erweitern
                        "currency_id": DEFAULT_CURRENCY_ID,
                        "currency_factor": DEFAULT_CURRENCY_FACTOR,
                    }
                ],
            }
            if ref_nr:
                payload["reference_nr"] = ref_nr

            r = requests.post(
                MANUAL_ENTRIES_V3,
                headers={**_auth(), "Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            if r.status_code == 401:
                refresh_access_token()
                r = requests.post(
                    MANUAL_ENTRIES_V3,
                    headers={**_auth(), "Content-Type": "application/json"},
                    json=payload,
                    timeout=30,
                )
            if r.status_code == 429:
                results.append({"row": idx + 1, "status": "Rate limited (429)"})
                continue

            r.raise_for_status()
            results.append({"row": idx + 1, "status": "OK", "id": r.json().get("id"), "reference_nr": ref_nr})
        except requests.HTTPError as e:
            try:
                err_txt = e.response.text
            except Exception:
                err_txt = str(e)
            results.append({"row": idx + 1, "status": f"HTTP {e.response.status_code}", "error": err_txt})
        except Exception as e:
            results.append({"row": idx + 1, "status": "ERROR", "error": str(e)})

    if not results:
        st.info("Keine gültigen Zeilen zum Posten gefunden.")
    else:
        st.success(f"Fertig. {sum(1 for r in results if r.get('status')=='OK')} Buchung(en) erfolgreich gepostet.")
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
