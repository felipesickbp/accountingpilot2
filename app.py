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

SCOPES = "openid profile email offline_access company_profile"

st.set_page_config(page_title="bexio Bulk Manual Entries (v3)", page_icon="📘")

if "oauth" not in st.session_state:
    st.session_state.oauth = {}
if "bulk_df" not in st.session_state:
    st.session_state.bulk_df = None


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
with st.expander("📒 Kontenplan – alle Konten & IDs"):
    try:
        # Attempt to retrieve a large page; if your tenant has more, add simple pagination as needed
        r = requests.get(f"{ACCOUNTS_V3}?limit=1000", headers=_auth(), timeout=30)
        r.raise_for_status()
        accounts = r.json() if isinstance(r.json(), list) else []
        if accounts:
            df_accts = pd.DataFrame([
                {
                    "id": a.get("id"),
                    "number": a.get("number"),
                    "name": a.get("name"),
                    "is_active": a.get("is_active"),
                    "type": a.get("type"),
                }
                for a in accounts
            ]).sort_values(["number", "id"], na_position="last")
            st.dataframe(df_accts, use_container_width=True, hide_index=True)
            st.caption("Nutze die Spalten ‘soll’/‘haben’ unten mit diesen **id**-Werten (nicht ‘number’).")
        else:
            st.info("Keine Konten gefunden.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Konten: {e}")

# =========================
# Bulk editor grid
# =========================

st.subheader("Mehrere Buchungen erfassen")
st.caption("Spalten: **buchungsnummer**, **datum**, **betrag**, **soll**, **haben**. ‘soll’/‘haben’ sind **account_id**-Werte.")

# Default currency settings (keine UI – simpel halten)
DEFAULT_CURRENCY_ID = 1  # CHF in den meisten Mandanten (anpassen falls nötig)
DEFAULT_CURRENCY_FACTOR = 1.0

# Create an initial empty grid if not present
if st.session_state.bulk_df is None:
    st.session_state.bulk_df = pd.DataFrame(
        {
            "buchungsnummer": ["" for _ in range(5)],
            "datum": [dt_date.today() for _ in range(5)],
            "betrag": [0.00 for _ in range(5)],
            "soll": [None for _ in range(5)],
            "haben": [None for _ in range(5)],
        }
    )

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

            debit_id = int(row.get("soll"))
            credit_id = int(row.get("haben"))
            if debit_id <= 0 or credit_id <= 0:
                raise ValueError("soll/haben (account_id) müssen > 0 sein.")

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
