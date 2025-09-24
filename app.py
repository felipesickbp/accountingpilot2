import os, time, base64
import streamlit as st
import requests
import zipfile
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

# v2 (read-only import of accounts)
API_V2 = "https://api.bexio.com/2.0"

SCOPES = "openid profile email offline_access company_profile"

st.set_page_config(page_title="bexio Bulk Manual Entries (v3)", page_icon="📘")

# ---- Session defaults (do this BEFORE any access) ----
if "oauth" not in st.session_state:
    st.session_state.oauth = {}
if "bulk_df" not in st.session_state:
    st.session_state.bulk_df = None
if "acct_map_by_number" not in st.session_state:
    st.session_state.acct_map_by_number = {}
if "acct_df" not in st.session_state:
    st.session_state.acct_df = None

# Default currency
DEFAULT_CURRENCY_ID = 1
DEFAULT_CURRENCY_FACTOR = 1.0

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

# v2 read-only auth + fetch
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

# ======================================================
# STEP 2 — Bankdatei importieren (CSV -> Vorschau)
# ======================================================
st.header("2) Bankdatei importieren (CSV)")
st.caption("Lade eine CSV hoch. Wähle die Spalten, mit denen wir das Bexio-Gitter füllen.")

if "bank_csv_df" not in st.session_state:
    st.session_state.bank_csv_df = None
if "bank_map" not in st.session_state:
    st.session_state.bank_map = {
        "buchungsnummer": None,
        "datum": None,
        "betrag": None,
        "soll": None,   # Kontonummer (oder id)
        "haben": None,  # Kontonummer (oder id)
    }

bank_file = st.file_uploader("Bank-CSV hochladen", type=["csv"])
col_enc1, col_enc2 = st.columns([2,1])
encoding = col_enc1.selectbox("Encoding", ["utf-8", "latin-1", "utf-16"], index=0)
decimal  = col_enc2.selectbox("Dezimaltrennzeichen", [".", ","], index=0)

def _try_read_csv(uploaded_file):
    raw = uploaded_file.getvalue()

    # If it's actually an Excel file (xlsx)
    if raw[:4] == b"PK\x03\x04":
        try:
            df = pd.read_excel(io.BytesIO(raw), dtype=str, keep_default_na=False)
            return df
        except Exception as e:
            raise ValueError(f"Excel-Datei konnte nicht gelesen werden: {e}")

    # Try multiple encodings and delimiters
    encodings  = [encoding, "utf-8-sig", "cp1252", "latin-1", "utf-16", "utf-16le", "utf-16be"]
    delimiters = [None, ";", ",", "\t", "|"]
    errors = []

    for enc in encodings:
        for sep in delimiters:
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    sep=sep, engine="python",
                    encoding=enc,
                    decimal=decimal,
                    dtype=str,
                    keep_default_na=False,
                )
                # Heuristic: must have at least 1 column
                if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
                    return df
            except Exception as e:
                errors.append(f"{enc}/{repr(sep)} → {e}")

    raise ValueError("CSV konnte nicht gelesen werden. Versuche: "
                     + "; ".join(errors[:4]) + (" …" if len(errors) > 4 else ""))
    
if bank_file is not None:
    try:
        st.session_state.bank_csv_df = _try_read_csv(bank_file)
        st.success(f"CSV geladen: {st.session_state.bank_csv_df.shape[0]} Zeilen, {st.session_state.bank_csv_df.shape[1]} Spalten")
        st.dataframe(st.session_state.bank_csv_df.head(50), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"CSV konnte nicht gelesen werden: {e}")

# Mapping-UI, falls CSV vorhanden
if st.session_state.bank_csv_df is not None and len(st.session_state.bank_csv_df.columns) > 0:
    st.subheader("Spalten zuordnen (CSV → Bexio-Felder)")
    cols = ["<keine>"] + list(st.session_state.bank_csv_df.columns)
    c1, c2 = st.columns(2)
    st.session_state.bank_map["buchungsnummer"] = c1.selectbox("buchungsnummer", options=cols, index=0)
    st.session_state.bank_map["datum"]          = c2.selectbox("datum", options=cols, index=0)
    c3, c4 = st.columns(2)
    st.session_state.bank_map["betrag"]         = c3.selectbox("betrag (positiv = Debit / negativ = Kredit, oder umgekehrt)", options=cols, index=0)
    st.session_state.bank_map["soll"]           = c4.selectbox("soll (Kontonummer oder account_id)", options=cols, index=0)
    st.session_state.bank_map["haben"]          = st.selectbox("haben (Kontonummer oder account_id)", options=cols, index=0)

    build_btn = st.button("Konvertieren → Bexio-Gitter")
else:
    build_btn = False

# ======================================================
# STEP 3 — Gitter befüllen, bearbeiten, posten
# ======================================================

def _parse_date_to_iso(x: str) -> str:
    s = (x or "").strip()
    if not s:
        return ""
    try:
        # Cleveres Parsen, erlaubt 01.02.2025, 1/2/25, etc.
        d = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(d):
            d = pd.to_datetime(s, dayfirst=False, errors="coerce")
        if pd.isna(d):
            return ""
        return d.date().isoformat()
    except Exception:
        return ""

def _to_float(x: str) -> float:
    if x is None: 
        return 0.0
    s = str(x).strip().replace("’","").replace("'","")
    # Versuche beide Dezimal-Formate
    try:
        return float(s.replace(" ", "").replace(",", "."))  # 1 234,56 -> 1234.56
    except Exception:
        try:
            return float(s)
        except Exception:
            return 0.0

# Create/edit bulk_df from mapping
if build_btn:
    src = st.session_state.bank_csv_df
    mp  = st.session_state.bank_map

    def pick(colname):
        sel = mp.get(colname)
        if sel and sel in src.columns:
            return src[sel]
        # Always return a Series (right length), not a scalar string
        return pd.Series([""] * len(src), index=src.index, dtype="string")


    df_new = pd.DataFrame({
        "buchungsnummer": pick("buchungsnummer"),
        "datum":          pick("datum").apply(_parse_date_to_iso),
        "betrag":         pick("betrag").apply(_to_float),
        "soll":           pick("soll").astype(str),
        "haben":          pick("haben").astype(str),
    })

    # Fallbacks/Defaults
    if "datum" in df_new and (df_new["datum"] == "").all():
        df_new["datum"] = dt_date.today().isoformat()

    st.session_state.bulk_df = df_new
    st.success(f"Gitter befüllt: {len(df_new)} Zeilen. Du kannst jetzt editieren & posten.")

# Ensure correct dtypes for the editor:
if st.session_state.bulk_df is None:
    st.session_state.bulk_df = pd.DataFrame(
        {"buchungsnummer": [], "datum": [], "betrag": [], "soll": [], "haben": []}
    )

# Coerce betrag to float dtype; empty/invalid -> NaN (shown as blank)
if "betrag" in st.session_state.bulk_df.columns:
    st.session_state.bulk_df["betrag"] = pd.to_numeric(
        st.session_state.bulk_df["betrag"], errors="coerce"
    )


# Editor im Formular (stabil, kein Input-Verlust)
st.subheader("3) Kontrolle & Import")
with st.form("bulk_entries_form", clear_on_submit=False):
    edited_df = st.data_editor(
        st.session_state.bulk_df if st.session_state.bulk_df is not None else pd.DataFrame(
            {"buchungsnummer": [], "datum": [], "betrag": [], "soll": [], "haben": []}
        ),
        key="bulk_grid",
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "buchungsnummer": st.column_config.TextColumn("buchungsnummer", help="Optional; leer = auto Ref-Nr."),
            "datum": st.column_config.TextColumn("datum (YYYY-MM-DD oder frei; wird geparst)"),
            "betrag": st.column_config.NumberColumn("betrag", min_value=0.0, step=0.05, format="%.2f"),
            "soll": st.column_config.TextColumn("soll (Kontonummer oder account_id)"),
            "haben": st.column_config.TextColumn("haben (Kontonummer oder account_id)"),
        }


    )
    colA, colB = st.columns(2)
    auto_ref   = colA.checkbox("Referenznummer automatisch beziehen (wenn leer)", value=True)
    submitted  = colB.form_submit_button("Buchungen posten", type="primary")

# Persist back
st.session_state.bulk_df = edited_df

# --- Kontonummer/id → account_id Auflösung ---
def resolve_account_id_from_number_or_id(val):
    if val is None:
        return None
    key = str(val).strip()
    if key == "":
        return None
    if key in st.session_state.acct_map_by_number:  # prefer Kontonummer
        return int(st.session_state.acct_map_by_number[key])
    try:
        return int(key)  # treat as raw account_id
    except Exception:
        return None

# --- POSTEN ---
if submitted:
    if edited_df is None or edited_df.empty:
        st.warning("Keine Zeilen im Gitter.")
    else:
        rows = edited_df.fillna("")
        results = []
        for idx, row in rows.iterrows():
            try:
                if (str(row.get("datum","")).strip() == "" and
                    float(row.get("betrag",0) or 0) == 0 and
                    str(row.get("soll","")).strip() == "" and
                    str(row.get("haben","")).strip() == ""):
                    continue

                # Datum
                if isinstance(row.get("datum"), dt_date):
                    date_iso = row["datum"].isoformat()
                else:
                    date_iso = _parse_date_to_iso(str(row.get("datum","")))
                if not date_iso:
                    raise ValueError(f"Ungültiges Datum in Zeile {idx+1}")

                amount = float(row.get("betrag") or 0)
                if amount <= 0:
                    raise ValueError(f"Betrag muss > 0 sein (Zeile {idx+1}).")

                debit_id  = resolve_account_id_from_number_or_id(row.get("soll"))
                credit_id = resolve_account_id_from_number_or_id(row.get("haben"))
                if not debit_id or not credit_id:
                    raise ValueError(f"Konto unbekannt (Zeile {idx+1}): bitte Kontonummern prüfen / Kontenplan importieren.")

                ref_nr = str(row.get("buchungsnummer") or "").strip()
                if auto_ref and not ref_nr:
                    rr = requests.get(NEXT_REF_V3, headers=_auth(), timeout=15)
                    rr.raise_for_status()
                    ref_nr = (rr.json() or {}).get("next_ref_nr") or ""

                payload = {
                    "type": "manual_single_entry",
                    "date": date_iso,
                    "entries": [{
                        "debit_account_id": int(debit_id),
                        "credit_account_id": int(credit_id),
                        "amount": float(amount),
                        "description": "",
                        "currency_id": int(DEFAULT_CURRENCY_ID),
                        "currency_factor": float(DEFAULT_CURRENCY_FACTOR),
                    }],
                }
                if ref_nr:
                    payload["reference_nr"] = ref_nr

                r = requests.post(
                    MANUAL_ENTRIES_V3,
                    headers={**_auth(), "Content-Type": "application/json"},
                    json=payload, timeout=30
                )
                if r.status_code == 401:
                    refresh_access_token()
                    r = requests.post(
                        MANUAL_ENTRIES_V3,
                        headers={**_auth(), "Content-Type": "application/json"},
                        json=payload, timeout=30
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
