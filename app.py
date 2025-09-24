import os, time, base64, io
import pandas as pd
import streamlit as st
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv
from datetime import date as dt_date

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

st.set_page_config(page_title="bexio Bulk Manual Entries (v3)", page_icon="📘", layout="wide")

# =========================
# SESSION DEFAULTS
# =========================
if "step" not in st.session_state:
    st.session_state.step = 1  # 1=Kontenplan, 2=Bankdatei, 3=Kontrolle & Import

if "oauth" not in st.session_state:
    st.session_state.oauth = {}

if "acct_map_by_number" not in st.session_state:
    st.session_state.acct_map_by_number = {}
if "acct_df" not in st.session_state:
    st.session_state.acct_df = None
if "selected_bank_number" not in st.session_state:
    st.session_state.selected_bank_number = None

if "bank_csv_df" not in st.session_state:
    st.session_state.bank_csv_df = None
if "bank_csv_view_df" not in st.session_state:
    st.session_state.bank_csv_view_df = None
if "bank_start_row" not in st.session_state:
    st.session_state.bank_start_row = 1
if "bank_map" not in st.session_state:
    st.session_state.bank_map = {
        "buchungsnummer": None,
        "datum": None,
        "betrag": None,
        "soll": None,   # Kontonummer (oder id)
        "haben": None,  # Kontonummer (oder id)
    }

if "bulk_df" not in st.session_state:
    st.session_state.bulk_df = None

# Defaults for posting
DEFAULT_CURRENCY_ID = 1
DEFAULT_CURRENCY_FACTOR = 1.0

# =========================
# AUTH HELPERS
# =========================
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

def _auth_v2():
    return {
        **auth_header(st.session_state.oauth["access_token"]),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

# =========================
# API HELPERS
# =========================
def fetch_all_accounts_v2(limit=2000):
    """Read-only Kontenplan via /2.0/accounts/search."""
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
# SMALL UTILITIES
# =========================
def _parse_date_to_iso(x: str) -> str:
    s = (str(x) if x is not None else "").strip()
    if not s:
        return ""
    try:
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
    try:
        return float(s.replace(" ", "").replace(",", "."))  # 1 234,56 -> 1234.56
    except Exception:
        try:
            return float(s)
        except Exception:
            return 0.0

def resolve_account_id_from_number_or_id(val):
    """Accept a Kontonummer (preferred) or a raw account_id and return account_id (int)."""
    if val is None:
        return None
    key = str(val).strip()
    if key == "":
        return None
    if key in st.session_state.acct_map_by_number:
        return int(st.session_state.acct_map_by_number[key])
    try:
        return int(key)
    except Exception:
        return None

def read_csv_or_excel(uploaded_file, encoding_preference: str, decimal: str) -> pd.DataFrame:
    """Robust CSV reader; also supports xlsx disguised as csv."""
    raw = uploaded_file.getvalue()
    # Excel magic number
    if raw[:4] == b"PK\x03\x04":
        df = pd.read_excel(io.BytesIO(raw), dtype=str, keep_default_na=False)
        return df

    encodings  = [encoding_preference, "utf-8-sig", "cp1252", "latin-1", "utf-16", "utf-16le", "utf-16be"]
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
                if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
                    return df
            except Exception as e:
                errors.append(f"{enc}/{repr(sep)} → {e}")
    raise ValueError("CSV konnte nicht gelesen werden. Versuche: "
                     + "; ".join(errors[:4]) + (" …" if len(errors) > 4 else ""))

# =========================
# LAYOUT: HEADER + WIZARD NAV
# =========================
st.title("📘 bexio Bulk Manual Entries (API v3)")
handle_callback()
if need_login():
    st.info("Verbinde dein bexio Konto, um Buchungen zu posten.")
    login_link()
    st.stop()
if time.time() > st.session_state.oauth.get("expires_at", 0):
    with st.spinner("Session wird erneuert …"):
        refresh_access_token()

col_nav1, col_nav2, col_nav3, col_reset = st.columns([1,1,1,1])
col_nav1.button("1) Kontenplan", on_click=lambda: st.session_state.update(step=1))
col_nav2.button("2) Bankdatei", on_click=lambda: st.session_state.update(step=2))
col_nav3.button("3) Kontrolle & Import", on_click=lambda: st.session_state.update(step=3))
if col_reset.button("🔁 Assistent zurücksetzen"):
    for k in ["acct_map_by_number","acct_df","selected_bank_number","bank_csv_df",
              "bank_csv_view_df","bulk_df","bank_map","bank_start_row"]:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.step = 1
    st.experimental_rerun()

st.markdown("---")

# =========================
# STEP 1 — KONTENPLAN
# =========================
if st.session_state.step == 1:
    st.subheader("1) Kontenplan aus bexio importieren (READ via v2)")
    st.caption("Nur zum Einlesen der Mapping-Tabelle (account_no → id). Postings bleiben über v3.")

    c1, c2 = st.columns([1,1])
    if c1.button("Kontenplan importieren"):
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

    if st.session_state.acct_df is not None and not st.session_state.acct_df.empty:
        st.dataframe(st.session_state.acct_df, use_container_width=True, hide_index=True)
        st.caption("Tabelle: **Kontonummer → externe ID** (id).")

        st.subheader("Bankkonto wählen (optional, für Auto-Zuordnung)")
        df = st.session_state.acct_df.copy()

        # Normalize for filtering
        df["number_str"] = df["number"].astype(str).str.strip()
        df["name_str"]   = df["name"].astype(str)
        # Heuristic for bank/cash accounts: account_no 10xx or name contains keywords (case-insensitive)
        is_bank_like = (
            df["number_str"].str.match(r"10\d{2}(-[A-Z])?$", na=False) |
            df["name_str"].str.contains(r"\b(bank|kasse|konto|cash)\b", case=False, regex=True, na=False)
        )
        active_mask = (df["active"] == True)

        banks_df = df[active_mask & is_bank_like].copy()
        if banks_df.empty:
            # Fallback: all active accounts
            banks_df = df[active_mask].copy()

        # Simple filter box to quickly find 1020 etc.
        filt = st.text_input("Kontensuche (Nummer/Name enthält …)", value="")
        if filt.strip():
            s = filt.strip().lower()
            banks_df = banks_df[
                banks_df["number_str"].str.lower().str.contains(s, na=False) |
                banks_df["name_str"].str.lower().str.contains(s, na=False)
            ]

        if not banks_df.empty:
            opts = [f"{r['number_str']} · {r['name_str']} (id {int(r['id'])})" for _, r in banks_df.iterrows()]
            choice = st.selectbox("Bankkonto (optional)", ["<keins>"] + opts, index=0)
            if choice != "<keins>":
                st.session_state.selected_bank_number = choice.split(" · ", 1)[0].strip()
                st.caption(
                    f"Gewählt: **{st.session_state.selected_bank_number}** – in Schritt 3 wird für **positive Beträge** automatisch **soll**, "
                    f"für **negative Beträge** automatisch **haben** mit diesem Konto vorbelegt (falls leer)."
                )
        else:
            st.info("Keine aktiven Konten passend zum Filter gefunden.")

    st.markdown("---")
    disabled_next = st.session_state.acct_df is None or st.session_state.acct_df.empty
    st.button("Weiter → 2) Bankdatei", disabled=disabled_next,
              on_click=lambda: st.session_state.update(step=2))

# =========================
# STEP 2 — BANKDATEI & MAPPING
# =========================
elif st.session_state.step == 2:
    if st.session_state.acct_df is None or st.session_state.acct_df.empty:
        st.warning("Bitte zuerst Schritt 1 abschließen (Kontenplan importieren).")
        st.stop()

    st.subheader("2) Bankdatei importieren (CSV/Excel)")
    st.caption("Lade eine CSV (oder Excel mit .csv-Endung). Danach Startzeile und Spalten zuordnen.")

    bank_file = st.file_uploader("Bank-CSV hochladen", type=["csv"])

    col_enc1, col_enc2 = st.columns([2,1])
    encoding = col_enc1.selectbox("Encoding", ["utf-8", "latin-1", "utf-16"], index=0)
    decimal  = col_enc2.selectbox("Dezimaltrennzeichen", [".", ","], index=0)

    if bank_file is not None:
        try:
            st.session_state.bank_csv_df = read_csv_or_excel(bank_file, encoding, decimal)

            st.markdown("**Ab welcher Zeile beginnen die Daten?** (1 = erste Zeile der Datei)")
            st.session_state.bank_start_row = st.number_input(
                "Startzeile (1-basiert)", min_value=1, value=int(st.session_state.bank_start_row), step=1
            )

            df_src = st.session_state.bank_csv_df
            df_view = df_src.iloc[st.session_state.bank_start_row - 1 :].copy()
            df_view.insert(0, "csv_row", df_view.index + 1)  # 1-based row numbers
            st.session_state.bank_csv_view_df = df_view

            st.success(f"Verwende {len(df_view)} Datenzeilen ab Zeile {st.session_state.bank_start_row}.")
            st.dataframe(df_view.head(50), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"CSV konnte nicht gelesen werden: {e}")

    src_for_mapping = st.session_state.get("bank_csv_view_df", None)
    if src_for_mapping is not None and not src_for_mapping.empty:
        st.subheader("Spalten zuordnen (CSV → Bexio-Felder)")
        cols = ["<keine>"] + [c for c in src_for_mapping.columns if c != "csv_row"]
        c1, c2 = st.columns(2)
        st.session_state.bank_map["buchungsnummer"] = c1.selectbox("buchungsnummer", options=cols, index=0)
        st.session_state.bank_map["datum"]          = c2.selectbox("datum", options=cols, index=0)
        c3, c4 = st.columns(2)
        st.session_state.bank_map["betrag"]         = c3.selectbox("betrag (positiv = Debit / negativ = Kredit)", options=cols, index=0)
        st.session_state.bank_map["soll"]           = c4.selectbox("soll (Kontonummer oder account_id)", options=cols, index=0)
        st.session_state.bank_map["haben"]          = st.selectbox("haben (Kontonummer oder account_id)", options=cols, index=0)

        def pick(src, colname):
            sel = st.session_state.bank_map.get(colname)
            if sel and sel in src.columns:
                return src[sel]
            return pd.Series([""] * len(src), index=src.index, dtype="string")

        if st.button("Konvertieren → Vorschau-Gitter"):
            src = src_for_mapping
            df_new = pd.DataFrame({
                "csv_row":        src["csv_row"],
                "buchungsnummer": pick(src, "buchungsnummer"),
                "datum":          pick(src, "datum").apply(_parse_date_to_iso),
                "betrag":         pick(src, "betrag").apply(_to_float),
                "soll":           pick(src, "soll").astype(str),
                "haben":          pick(src, "haben").astype(str),
            })
            if (df_new["datum"] == "").all():
                df_new["datum"] = dt_date.today().isoformat()

            # Auto-assign bank account to soll/haben by sign (only if empty cells)
            if st.session_state.selected_bank_number:
                pos_mask = df_new["betrag"].fillna(0) > 0
                neg_mask = df_new["betrag"].fillna(0) < 0
                df_new.loc[pos_mask & (df_new["soll"].str.strip() == ""),  "soll"]  = st.session_state.selected_bank_number
                df_new.loc[neg_mask & (df_new["haben"].str.strip() == ""), "haben"] = st.session_state.selected_bank_number

            # Coerce amount to numeric for editor
            df_new["betrag"] = pd.to_numeric(df_new["betrag"], errors="coerce")

            st.session_state.bulk_df = df_new
            st.success(f"Gitter erstellt: {len(df_new)} Zeilen.")

    st.markdown("---")
    disabled_next = st.session_state.bulk_df is None or st.session_state.bulk_df.empty
    st.button("Weiter → 3) Kontrolle & Import", disabled=disabled_next,
              on_click=lambda: st.session_state.update(step=3))

# =========================
# STEP 3 — KONTROLLE & IMPORT
# =========================
elif st.session_state.step == 3:
    if st.session_state.bulk_df is None or st.session_state.bulk_df.empty:
        st.warning("Bitte zuerst Schritt 2 abschließen (Vorschau-Gitter erzeugen).")
        st.stop()

    st.subheader("3) Kontrolle & Import")

    with st.form("bulk_entries_form", clear_on_submit=False):
        edited_df = st.data_editor(
            st.session_state.bulk_df,
            key="bulk_grid",
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "csv_row":        st.column_config.TextColumn("csv_row (aus CSV)", help="Ursprüngliche Zeilennummer."),
                "buchungsnummer": st.column_config.TextColumn("buchungsnummer", help="Optional; leer = auto Ref-Nr."),
                "datum":          st.column_config.TextColumn("datum (YYYY-MM-DD oder frei; wird geparst)"),
                "betrag":         st.column_config.NumberColumn("betrag", min_value=0.0, step=0.05, format="%.2f"),
                "soll":           st.column_config.TextColumn("soll (Kontonummer oder account_id)"),
                "haben":          st.column_config.TextColumn("haben (Kontonummer oder account_id)"),
            }
        )
        colA, colB = st.columns(2)
        auto_ref   = colA.checkbox("Referenznummer automatisch beziehen (wenn leer)", value=True)
        submitted  = colB.form_submit_button("Buchungen posten", type="primary")

    # Persist back
    st.session_state.bulk_df = edited_df

    if submitted:
        if edited_df is None or edited_df.empty:
            st.warning("Keine Zeilen im Gitter.")
        else:
            rows = edited_df.fillna("")
            results = []
            for idx, row in rows.iterrows():
                try:
                    # Skip completely empty lines
                    if (str(row.get("datum","")).strip() == "" and
                        float(row.get("betrag",0) or 0) == 0 and
                        str(row.get("soll","")).strip() == "" and
                        str(row.get("haben","")).strip() == ""):
                        continue

                    # Date parsing
                    if isinstance(row.get("datum"), dt_date):
                        date_iso = row["datum"].isoformat()
                    else:
                        date_iso = _parse_date_to_iso(str(row.get("datum","")))
                    if not date_iso:
                        raise ValueError(f"Ungültiges Datum in Zeile {idx+1}")

                    # Amount: API expects positive value; side is by debit/credit
                    amount_raw = float(row.get("betrag") or 0)
                    if amount_raw == 0:
                        raise ValueError(f"Betrag darf nicht 0 sein (Zeile {idx+1}).")
                    amount = abs(amount_raw)

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
                        results.append({"row": idx + 1, "csv_row": row.get("csv_row",""), "status": "Rate limited (429)"})
                        continue

                    r.raise_for_status()
                    results.append({"row": idx + 1, "csv_row": row.get("csv_row",""), "status": "OK", "id": r.json().get("id"), "reference_nr": ref_nr})
                except requests.HTTPError as e:
                    try:
                        err_txt = e.response.text
                    except Exception:
                        err_txt = str(e)
                    results.append({"row": idx + 1, "csv_row": row.get("csv_row",""), "status": f"HTTP {e.response.status_code}", "error": err_txt})
                except Exception as e:
                    results.append({"row": idx + 1, "csv_row": row.get("csv_row",""), "status": "ERROR", "error": str(e)})

            if not results:
                st.info("Keine gültigen Zeilen zum Posten gefunden.")
            else:
                st.success(f"Fertig. {sum(1 for r in results if r.get('status')=='OK')} Buchung(en) erfolgreich gepostet.")
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
