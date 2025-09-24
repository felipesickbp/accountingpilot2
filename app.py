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
# SCHEMA & HELPERS
# =========================

REQUIRED_COLS = ["csv_row", "buchungsnummer", "datum", "betrag", "soll", "haben", "beschreibung"]

# Load CSS (call this right after set_page_config)
def _inject_local_css(file_path: str = "styles.css"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Custom CSS konnte nicht geladen werden: {e}")

_inject_local_css()


def ensure_schema(df_in: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with exactly the REQUIRED_COLS and safe dtypes for data_editor."""
    # If input is None, start from an empty frame
    if df_in is None:
        df_in = pd.DataFrame()

    df = pd.DataFrame(columns=REQUIRED_COLS)
    for c in REQUIRED_COLS:
        df[c] = df_in[c] if c in df_in.columns else ""

    # dtypes preferred by editor
    try:
        df["csv_row"] = pd.to_numeric(df["csv_row"], errors="coerce").astype("Int64")
    except Exception:
        df["csv_row"] = pd.Series([pd.NA]*len(df), dtype="Int64")

    df["betrag"] = pd.to_numeric(df["betrag"], errors="coerce").astype(float)

    for c in ["buchungsnummer", "datum", "soll", "haben", "beschreibung"]:
        # keep as string dtype for editor stability
        df[c] = df[c].astype(str)

    return df


def _parse_date_to_iso(x: str) -> str:
    s = (str(x) if x is not None else "").strip()
    if not s:
        return ""
    d = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if pd.isna(d):
        d = pd.to_datetime(s, dayfirst=False, errors="coerce")
    return "" if pd.isna(d) else d.date().isoformat()

def _to_float(x) -> float:
    if x is None: return 0.0
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

def _make_unique_columns(cols, prefix="col"):
    """
    Return a list of unique, non-empty column names.
    - Trims whitespace
    - Replaces empty/unnamed with prefix_1, prefix_2, ...
    - Deduplicates by adding suffixes: "Name", "Name__2", "Name__3", ...
    """
    out = []
    seen = {}
    for i, c in enumerate(cols, start=1):
        name = str(c).strip()
        if name == "" or name.lower().startswith("unnamed:"):
            name = f"{prefix}_{i}"
        base = name
        if base in seen:
            seen[base] += 1
            name = f"{base}__{seen[base]}"
        else:
            seen[base] = 1
        out.append(name)
    return out

def read_csv_or_excel(uploaded_file, encoding_preference: str, decimal: str) -> pd.DataFrame:
    """
    Robust reader for CSV or Excel (xlsx disguised as csv).
    - Ensures unique, non-empty column names
    - Reserves 'csv_row' name (renames existing to 'csv_row_file')
    - Tries multiple encodings and delimiters
    - If header parsing fails, falls back to header=None and synthesizes column names
    Returns a pandas DataFrame (never None) or raises ValueError.
    """
    raw = uploaded_file.getvalue()

    # Detect Excel by magic number
    if len(raw) >= 4 and raw[:4] == b"PK\x03\x04":
        try:
            df = pd.read_excel(io.BytesIO(raw), dtype=str, keep_default_na=False)
        except Exception as e:
            raise ValueError(f"Excel-Datei konnte nicht gelesen werden: {e}")

        df.columns = _make_unique_columns(df.columns, prefix="col")
        # Reserve 'csv_row'
        if "csv_row" in df.columns:
            df = df.rename(columns={"csv_row": "csv_row_file"})
        return df

    # Try CSV with various encodings and delimiters
    encodings  = [encoding_preference, "utf-8-sig", "cp1252", "latin-1", "utf-16", "utf-16le", "utf-16be"]
    delimiters = [None, ";", ",", "\t", "|"]
    errors = []

    for enc in encodings:
        for sep in delimiters:
            # 1) Try with header row
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    sep=sep,
                    engine="python",
                    encoding=enc,
                    decimal=decimal,
                    dtype=str,
                    keep_default_na=False,
                    header=0,
                )
                if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
                    df.columns = _make_unique_columns(df.columns, prefix="col")
                    if "csv_row" in df.columns:
                        df = df.rename(columns={"csv_row": "csv_row_file"})
                    return df
            except Exception as e:
                errors.append(f"header=0 {enc}/{repr(sep)} → {e}")

            # 2) Fallback: no header in file (treat first row as data)
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    sep=sep,
                    engine="python",
                    encoding=enc,
                    decimal=decimal,
                    dtype=str,
                    keep_default_na=False,
                    header=None,
                )
                if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
                    # Synthesize column names
                    df.columns = _make_unique_columns([f"col_{i+1}" for i in range(df.shape[1])], prefix="col")
                    if "csv_row" in df.columns:
                        df = df.rename(columns={"csv_row": "csv_row_file"})
                    return df
            except Exception as e:
                errors.append(f"header=None {enc}/{repr(sep)} → {e}")

    # If we got here, nothing worked
    msg = "CSV konnte nicht gelesen werden. Versuche: " + "; ".join(errors[:6])
    if len(errors) > 6:
        msg += " …"
    raise ValueError(msg)


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
        "datum": None,
        "betrag": None,
        "beschreibung": None,  # only these three are mapped by the user
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
# LAYOUT: HEADER + WIZARD NAV
# =========================
st.title("📘 Bexio Bulk Manual Entries")
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
    st.rerun()

st.markdown("---")

# =========================
# STEP 1 — KONTENPLAN
# =========================
if st.session_state.step == 1:
    st.subheader("1) Kontenplan aus bexio importieren")
    st.caption("Nur zum Einlesen der Mapping-Tabelle.")

    c1, _ = st.columns([1,1])
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
        # Heuristic for bank/cash accounts: 10xx or keywords
        is_bank_like = (
            df["number_str"].str.match(r"10\d{2}(-[A-Z])?$", na=False) |
            df["name_str"].str.contains(r"\b(bank|kasse|konto|cash)\b", case=False, regex=True, na=False)
        )
        active_mask = (df["active"] == True)
        banks_df = df[active_mask & is_bank_like].copy()
        if banks_df.empty:
            # Fallback: all active accounts
            banks_df = df[active_mask].copy()

      

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

    # --- robust read + preview ---
    if bank_file is not None:
        try:
            st.session_state.bank_csv_df = read_csv_or_excel(bank_file, encoding, decimal)
            df_src = st.session_state.bank_csv_df

            if not isinstance(df_src, pd.DataFrame) or df_src.empty:
                raise ValueError("Datei enthält keine Datenzeilen.")

            st.markdown("**Ab welcher Zeile beginnen die Daten?** (1 = erste Zeile der Datei)")
            st.session_state.bank_start_row = st.number_input(
                "Startzeile (1-basiert)", min_value=1, value=int(st.session_state.bank_start_row), step=1
            )

            start_idx = max(0, int(st.session_state.bank_start_row) - 1)
            if start_idx >= len(df_src):
                raise ValueError("Startzeile liegt hinter dem Dateiende.")

            # Build safe view and reserve our own csv_row
            df_view = df_src.iloc[start_idx:].copy()
            if "csv_row" in df_view.columns:
                df_view = df_view.rename(columns={"csv_row": "csv_row_file"})
            df_view.insert(0, "csv_row", df_view.index + 1)  # 1-based original row no.
            df_view = df_view.reset_index(drop=True)

            st.session_state.bank_csv_view_df = df_view

            st.success(f"Verwende {len(df_view)} Datenzeilen ab Zeile {st.session_state.bank_start_row}.")
            st.dataframe(df_view.head(50), use_container_width=True, hide_index=True)

        except Exception as e:
            st.session_state.bank_csv_view_df = None
            st.error(f"CSV konnte nicht gelesen werden: {e}")

    src_for_mapping = st.session_state.get("bank_csv_view_df", None)

    # --- mapping UI only when we truly have rows ---
if isinstance(src_for_mapping, pd.DataFrame) and not src_for_mapping.empty:
    st.subheader("Spalten zuordnen (CSV → Bexio-Felder)")

    # Only these are mapped by the user (all headers except synthetic csv_row)
    cols = ["<keine>"] + [c for c in src_for_mapping.columns if c != "csv_row"]

    c1, c2 = st.columns(2)
    st.session_state.bank_map["datum"]  = c1.selectbox("datum", options=cols, index=0)
    st.session_state.bank_map["betrag"] = c2.selectbox("betrag (positiv = Debit / negativ = Kredit)", options=cols, index=0)

    c3, _ = st.columns([2,1])
    st.session_state.bank_map["beschreibung"] = c3.selectbox("beschreibung (Beschreibung / Text)", options=cols, index=0)

    # Safe picker: always returns a Series matching src_for_mapping length
    def pick(key: str) -> pd.Series:
        sel = st.session_state.bank_map.get(key)
        if sel and sel in src_for_mapping.columns:
            return src_for_mapping[sel]
        return pd.Series([""] * len(src_for_mapping), index=src_for_mapping.index, dtype="string")

    def convert_to_grid_and_advance():
        src = src_for_mapping

        df_new = pd.DataFrame({
            "csv_row":        src["csv_row"],                        # keep for traceability (hidden later)
            "buchungsnummer": "",                                    # left blank (auto-ref later)
            "datum":          pick("datum").apply(_parse_date_to_iso),
            "beschreibung":   pick("beschreibung").astype(str),      # normalized name early
            "betrag":         pick("betrag").apply(_to_float),
            "soll":           "",                                    # auto-fill by bank/sign if set
            "haben":          "",
        })

        # Default date if parsing failed everywhere
        if (df_new["datum"] == "").all():
            df_new["datum"] = dt_date.today().isoformat()

        # Assign bank account to soll/haben by sign (only if empty in target)
        if st.session_state.selected_bank_number:
            pos_mask = pd.to_numeric(df_new["betrag"], errors="coerce").fillna(0) > 0
            neg_mask = pd.to_numeric(df_new["betrag"], errors="coerce").fillna(0) < 0
            df_new.loc[pos_mask & (df_new["soll"].str.strip() == ""),  "soll"]  = st.session_state.selected_bank_number
            df_new.loc[neg_mask & (df_new["haben"].str.strip() == ""), "haben"] = st.session_state.selected_bank_number

        # Amount must be positive; side is encoded by debit/credit
        df_new["betrag"] = pd.to_numeric(df_new["betrag"], errors="coerce").abs()

        # Normalize schema/dtypes (keeps csv_row as Int64, betrag float, others str)
        st.session_state.bulk_df = ensure_schema(df_new)

        # Advance to step 3
        st.session_state.step = 3
        st.rerun()

    # Single button that converts AND advances
    st.button("Weiter → 3) Kontrolle & Import", type="primary", on_click=convert_to_grid_and_advance)
else:
    st.info("Lade eine Datei und wähle eine gültige Startzeile, um fortzufahren.")



# =========================
# STEP 3 — KONTROLLE & IMPORT
# =========================
elif st.session_state.step == 3:
    if st.session_state.bulk_df is None or st.session_state.bulk_df.empty:
        st.warning("Bitte zuerst Schritt 2 abschließen (Vorschau-Gitter erzeugen).")
        st.stop()

    st.subheader("3) Kontrolle & Import")

# Only the 6 fields Bexio needs in the editor
EDIT_COLS = ["buchungsnummer", "datum", "beschreibung", "betrag", "soll", "haben"]

with st.form("bulk_entries_form", clear_on_submit=False):
    # Pass only the visible columns to the editor
    edited_view = st.data_editor(
        st.session_state.bulk_df[EDIT_COLS],
        key="bulk_grid",
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "buchungsnummer": st.column_config.TextColumn("number (auto if empty)"),
            "datum":          st.column_config.TextColumn("date (YYYY-MM-DD; flexible parsing)"),
            "beschreibung":   st.column_config.TextColumn("label / description"),
            "betrag":         st.column_config.NumberColumn("amount", min_value=0.0, step=0.05, format="%.2f"),
            "soll":           st.column_config.TextColumn("debit (Kontonummer or account_id)"),
            "haben":          st.column_config.TextColumn("credit (Kontonummer or account_id)"),
        }
    )
    colA, colB = st.columns(2)
    auto_ref   = colA.checkbox("Referenznummer automatisch beziehen (wenn leer)", value=True)
    submitted  = colB.form_submit_button("Buchungen posten", type="primary")

# Write edited values back into the full DF (csv_row stays intact and hidden)
st.session_state.bulk_df.loc[:, EDIT_COLS] = edited_view

# Ensure schema/dtypes again (keeps csv_row as Int64, betrag as float, others as str)
st.session_state.bulk_df = ensure_schema(st.session_state.bulk_df)

if submitted:
    # Do NOT use .fillna("") on mixed dtypes (breaks Int64); use safe accessors instead
    rows = st.session_state.bulk_df.copy()

    if rows.empty:
        st.warning("Keine Zeilen im Gitter.")
    else:
        results = []

        for idx, row in rows.iterrows():
            try:
                # Safe getters that handle <NA> / None
                def _s(x):
                    try:
                        if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
                            return ""
                    except Exception:
                        pass
                    return str(x)

                def _f(x):
                    try:
                        return float(x)
                    except Exception:
                        return 0.0

                # Skip completely empty lines
                if (_s(row.get("datum")) == "" and
                    _f(row.get("betrag")) == 0 and
                    _s(row.get("soll")) == "" and
                    _s(row.get("haben")) == ""):
                    continue

                # Date parsing (flexible)
                date_val = row.get("datum")
                if isinstance(date_val, dt_date):
                    date_iso = date_val.isoformat()
                else:
                    date_iso = _parse_date_to_iso(_s(date_val))
                if not date_iso:
                    raise ValueError(f"Ungültiges Datum in Editor-Zeile {idx+1}")

                # Amount: already absolute from Step 2; enforce again
                amount = abs(_f(row.get("betrag")))
                if amount == 0:
                    raise ValueError(f"Betrag darf nicht 0 sein (Editor-Zeile {idx+1}).")

                # Accounts
                debit_id  = resolve_account_id_from_number_or_id(row.get("soll"))
                credit_id = resolve_account_id_from_number_or_id(row.get("haben"))
                if not debit_id or not credit_id:
                    raise ValueError(f"Konto unbekannt (Editor-Zeile {idx+1}): Kontonummern prüfen / Kontenplan importieren.")

                # Reference number
                ref_nr = _s(row.get("buchungsnummer"))
                if auto_ref and not ref_nr:
                    rr = requests.get(NEXT_REF_V3, headers=_auth(), timeout=15)
                    rr.raise_for_status()
                    ref_nr = (rr.json() or {}).get("next_ref_nr") or ""

                # Description
                desc = _s(row.get("beschreibung"))

                payload = {
                    "type": "manual_single_entry",
                    "date": date_iso,
                    "entries": [{
                        "debit_account_id": int(debit_id),
                        "credit_account_id": int(credit_id),
                        "amount": float(amount),
                        "description": desc,
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
                results.append({"row": idx + 1, "csv_row": row.ge_

