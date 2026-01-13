import os, time, base64, io
import pandas as pd
import streamlit as st
import requests
import math, random
from urllib.parse import urlencode
from dotenv import load_dotenv
from datetime import date as dt_date
import re
from pathlib import Path




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
API_V2 = "https://api.bexio.com/2.0"

SCOPES = "openid profile email offline_access company_profile"

st.set_page_config(page_title="Accounting Copilot (V 4.0)", page_icon="📘", layout="wide")
def ui_shell():
    st.markdown(
        """
        <style>
        /* Hide Streamlit default chrome a bit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {display: none;}

        /* General spacing */
        .block-container { padding-top: 2.0rem; padding-bottom: 2.5rem; }

        /* Card helper */
        .card {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 16px;
            padding: 18px 18px;
            background: rgba(255,255,255,0.02);
        }
        .muted { opacity: 0.8; }
        .big-title { font-size: 2.0rem; font-weight: 700; margin: 0 0 0.3rem 0; }
        .subtle { opacity: 0.85; margin-top: 0.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

ui_shell()


LOGO_PATH = Path("assets/logo.webp")

def render_header():
    left, right = st.columns([1, 8])

    with left:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=46)  # adjust width as you like
        else:
            st.caption("logo missing")

    with right:
        st.markdown(
            "<div style='margin-top:2px; font-size:2.0rem; font-weight:700;'>Accounting Copilot</div>",
            unsafe_allow_html=True
        )


def render_login_page():
    login_url = make_login_url()

    # add this block
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=64)

    st.markdown(
        f"""
        <div class="login-wrap">
          <div class="login-title">🤖 Accounting Copilot</div>
          ...
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# THEME CSS (optional)
# =========================
def _inject_local_css(file_path: str = "styles.css"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

_inject_local_css()

# =========================
# SCHEMA & HELPERS
# =========================

def _ensure_currency_code_map():
    if st.session_state.get("currency_code_to_id"):
        return

    mp = {}

    def _ingest(items):
        for c in items or []:
            try:
                cid = int(c.get("id"))
            except Exception:
                continue

            # bexio currency create uses field 'name' like 'JPY' (ISO) in many clients
            code = (
                c.get("name")
                or c.get("code")
                or c.get("currency_code")
                or c.get("short_name")
            )
            if code:
                mp.setdefault(str(code).upper().strip(), cid)

    endpoints = [
        f"{API_V3}/currencies",
        f"{API_V3}/accounting/currencies",
        f"{API_V2}/currencies",
    ]

    for url in endpoints:
        try:
            r = requests.get(url, headers=_auth() if "/3.0" in url else _auth_v2(), timeout=30)
            if r.status_code == 401:
                refresh_access_token()
                r = requests.get(url, headers=_auth() if "/3.0" in url else _auth_v2(), timeout=30)
            if r.status_code < 400:
                data = r.json()
                if isinstance(data, list):
                    _ingest(data)
                if mp:
                    break
        except Exception:
            pass

    st.session_state.currency_code_to_id = mp


def _currency_id_from_input(val: str | int | None) -> int | None:
    s = ("" if val is None else str(val)).strip()
    if not s:
        return int(DEFAULT_CURRENCY_ID)

    # numeric currency_id allowed
    try:
        return int(float(s))
    except Exception:
        pass

    code = s.upper()

    # env override: BEXIO_CURRENCY_ID_EUR=2, etc.
    env_key = f"BEXIO_CURRENCY_ID_{code}"
    env_val = os.getenv(env_key)
    if env_val:
        try:
            return int(env_val)
        except Exception:
            pass

    _ensure_currency_code_map()
    return st.session_state.get("currency_code_to_id", {}).get(code)


# VAT helpers and defaults
VAT_CODE_TO_RATE = {
    "UN81": 0.081,  # Standard (8.1%)
    "UR26": 0.026,  # Reduced
    "US38": 0.038,  # Special
}
DEFAULT_VAT_INPUT_ACCOUNT_NO  = "1170"  # Vorsteuer
DEFAULT_VAT_OUTPUT_ACCOUNT_NO = "2201"  # Umsatzsteuer (geschuldete MWST)

def _parse_vat_rate(val) -> float | None:
    """Accepts codes (UN81, UR26, …) or numeric (0.081 or 8.1 or '8.1%')."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    up = s.upper()
    if up in VAT_CODE_TO_RATE:
        return VAT_CODE_TO_RATE[up]
    try:
        x = float(s.replace("%", "").replace(",", "."))
        return (x / 100.0) if x > 1 else x
    except Exception:
        return None

# Extended schema: add the two new columns at the END (as requested)
REQUIRED_COLS = [
    "csv_row", "buchungsnummer", "datum", "betrag", "soll", "haben", "beschreibung",
    "mwst_code", "mwst_konto",
    "currency", "exchange_rate",
]

def ensure_schema(df_in: pd.DataFrame | None) -> pd.DataFrame:
    if df_in is None:
        df_in = pd.DataFrame()
    df = pd.DataFrame(columns=REQUIRED_COLS)
    for c in REQUIRED_COLS:
        df[c] = df_in[c] if c in df_in.columns else ""

    # dtypes that play nicely with st.data_editor
    try:
        df["csv_row"] = pd.to_numeric(df["csv_row"], errors="coerce").astype("Int64")
    except Exception:
        df["csv_row"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    df["betrag"] = pd.to_numeric(df["betrag"], errors="coerce").astype(float)

    # NEW
    df["exchange_rate"] = pd.to_numeric(df["exchange_rate"], errors="coerce").astype(float)
    df["currency"] = df["currency"].replace({None: ""}).fillna("").astype(str)

    for c in ["buchungsnummer", "datum", "soll", "haben", "beschreibung", "mwst_code", "mwst_konto"]:
        df[c] = df[c].replace({None: ""}).fillna("")
        df[c] = df[c].astype(str)
        df[c] = df[c].replace({"None": "", "nan": "", "<NA>": ""})

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
    if x is None:
        return 0.0
    s = str(x).strip().replace("’", "").replace("'", "")
    try:
        return float(s.replace(" ", "").replace(",", "."))
    except Exception:
        try:
            return float(s)
        except Exception:
            return 0.0

def resolve_account_id_from_number_or_id(val):
    """
    Accepts:
      - Kontonummer like "1020", "1020-A", "10 20", "1'020"
      - raw account_id like "12345"
    Returns account_id (int) or None.
    """
    if val is None:
        return None
    raw = str(val).strip()
    if raw == "":
        return None

    # Quick path: exact key
    if raw in st.session_state.acct_map_by_number:
        try:
            return int(st.session_state.acct_map_by_number[raw])
        except Exception:
            pass

    # Normalize possible kontonummer formats
    norm = raw.split("-", 1)[0]
    norm = norm.replace("’", "").replace("'", "").replace(" ", "")

    if norm in st.session_state.acct_map_by_number:
        try:
            return int(st.session_state.acct_map_by_number[norm])
        except Exception:
            pass

    # If it's an integer, assume it's already an account_id
    try:
        return int(norm)
    except Exception:
        return None


def post_with_backoff(url, headers, payload, max_retries=5, base_sleep=0.8):
    """POST with exponential backoff on 429/5xx. Returns (ok: bool, response_or_text)."""
    for attempt in range(max_retries + 1):
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        # Auto refresh once on 401
        if r.status_code == 401 and attempt == 0:
            refresh_access_token()
            r = requests.post(url, headers=headers, json=payload, timeout=30)

        if r.status_code < 400:
            return True, r

        if r.status_code not in (429, 500, 502, 503, 504) or attempt == max_retries:
            try:
                return False, f"HTTP {r.status_code}: {r.text}"
            except Exception:
                return False, f"HTTP {r.status_code}"

        sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.25)
        time.sleep(sleep_s)

    return False, "Max retries exceeded"

def apply_keyword_rules_to_df(df: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    """Fill soll/haben based on keyword rules. Does NOT overwrite existing non-empty cells."""
    if df is None or df.empty or not rules:
        return df
    out = df.copy()
    for col in ["beschreibung", "soll", "haben"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str)

    amt = pd.to_numeric(out.get("betrag", 0), errors="coerce").fillna(0)

    for r in rules:
        kw = str(r.get("keyword", "")).strip()
        acct = str(r.get("account", "")).strip()
        side = str(r.get("side", "haben")).strip().lower()
        if not kw or not acct:
            continue

        mask_kw = out["beschreibung"].str.contains(kw, case=False, na=False)

        if side == "soll":
            empty = out["soll"].str.strip() == ""
            out.loc[mask_kw & empty, "soll"] = acct
        elif side == "haben":
            empty = out["haben"].str.strip() == ""
            out.loc[mask_kw & empty, "haben"] = acct
        else:
            empty_s = out["soll"].str.strip() == ""
            empty_h = out["haben"].str.strip() == ""
            out.loc[mask_kw & (amt > 0) & empty_s, "soll"] = acct
            out.loc[mask_kw & (amt < 0) & empty_h, "haben"] = acct

    return out


def _make_unique_columns(cols, prefix="col"):
    out, seen = [], {}
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
        if "csv_row" in df.columns:
            df = df.rename(columns={"csv_row": "csv_row_file"})
        return df

    # CSV path
    encodings  = [encoding_preference, "utf-8-sig", "cp1252", "latin-1", "utf-16", "utf-16le", "utf-16be"]
    delimiters = [None, ";", ",", "\t", "|"]
    errors = []

    for enc in encodings:
        for sep in delimiters:
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    sep=sep, engine="python", encoding=enc, decimal=decimal,
                    dtype=str, keep_default_na=False, header=0
                )
                if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
                    df.columns = _make_unique_columns(df.columns, prefix="col")
                    if "csv_row" in df.columns:
                        df = df.rename(columns={"csv_row": "csv_row_file"})
                    return df
            except Exception as e:
                errors.append(f"header=0 {enc}/{repr(sep)} → {e}")

            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    sep=sep, engine="python", encoding=enc, decimal=decimal,
                    dtype=str, keep_default_na=False, header=None
                )
                if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
                    df.columns = _make_unique_columns([f"col_{i+1}" for i in range(df.shape[1])], prefix="col")
                    if "csv_row" in df.columns:
                        df = df.rename(columns={"csv_row": "csv_row_file"})
                    return df
            except Exception as e:
                errors.append(f"header=None {enc}/{repr(sep)} → {e}")

    msg = "CSV konnte nicht gelesen werden: " + "; ".join(errors[:6])
    if len(errors) > 6:
        msg += " …"
    raise ValueError(msg)

# =========================
# SESSION DEFAULTS
# =========================
if "step" not in st.session_state:
    st.session_state.step = 1
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
    st.session_state.bank_map = {"datum": None, "betrag": None, "beschreibung": None}
if "bulk_df" not in st.session_state:
    st.session_state.bulk_df = None

DEFAULT_CURRENCY_CODE = "CHF"
DEFAULT_CURRENCY_ID = 1
DEFAULT_CURRENCY_FACTOR = 1.0

# =========================
# SIDEBAR NAV
# =========================
STEP_LABELS = {
    1: "1) Kontenplan",
    2: "2) Bankdatei",
    3: "3) Kontrolle & Import",
}

def sidebar_nav():
    st.sidebar.markdown("### Navigation")
    step_names = [STEP_LABELS[1], STEP_LABELS[2], STEP_LABELS[3]]
    current_name = STEP_LABELS.get(st.session_state.step, STEP_LABELS[1])

    chosen = st.sidebar.radio(" ", step_names, index=step_names.index(current_name))

    new_step = {v: k for k, v in STEP_LABELS.items()}[chosen]
    if new_step != st.session_state.step:
        st.session_state.step = new_step
        st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("🔁 Assistent zurücksetzen", use_container_width=True):
        for k in ["acct_map_by_number","acct_df","selected_bank_number","bank_csv_df",
                  "bank_csv_view_df","bulk_df","bank_map","bank_start_row","bulk_grid"]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.step = 1
        st.rerun()

# =========================
# AUTH HELPERS
# =========================
def make_login_url():
    state = "anti-csrf-" + base64.urlsafe_b64encode(os.urandom(12)).decode("utf-8")
    params = {
        "client_id": BEXIO_CLIENT_ID,
        "redirect_uri": BEXIO_REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
    }
    return f"{AUTH_URL}?{urlencode(params)}"

def render_login_page():
    login_url = make_login_url()
    st.markdown(
        f"""
        <div class="login-wrap">
          <div class="login-title">🤖 Accounting Copilot</div>
          <div class="login-sub">
            Verbinde dein bexio Konto, um Banktransaktionen schnell als Buchungen zu posten (inkl. MWST).
          </div>

          <a class="cta" href="{login_url}" target="_blank" rel="noopener noreferrer">
            🔐 Mit bexio anmelden
          </a>

          <div class="small-hint">
            Falls der Button nicht reagiert: <a href="{login_url}" target="_blank" rel="noopener noreferrer">Login-Link öffnen</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

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
    qp = st.query_params
    code = qp.get("code")
    if not code:
        return

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": BEXIO_REDIRECT_URI,
        "client_id": BEXIO_CLIENT_ID,
        "client_secret": BEXIO_CLIENT_SECRET,
    }

    try:
        r = requests.post(TOKEN_URL, data=data, timeout=30)
    except requests.RequestException as e:
        st.error(f"Token request failed to send: {e}")
        st.query_params.clear()
        st.stop()

    if r.status_code >= 400:
        try:
            body = r.json()
        except Exception:
            body = r.text

        st.error(
            f"Token exchange failed ({r.status_code}).\n\n"
            f"Response:\n{body}"
        )

        with st.expander("Troubleshooting tips", expanded=True):
            st.markdown(
                "- **Redirect URI mismatch**: `BEXIO_REDIRECT_URI` must match **exactly** what’s configured in Bexio (scheme, host, path).\n"
                "- **Code already used or expired**: Try clicking *Sign in with bexio* again.\n"
                "- **Client credentials**: Ensure `BEXIO_CLIENT_ID` / `BEXIO_CLIENT_SECRET` are correct for this app.\n"
                "- **Wrong realm / token URL**: Double-check `TOKEN_URL`.\n"
                "- **Scope issues**: The requested `SCOPES` must be allowed for your app."
            )

        st.query_params.clear()
        st.stop()

    try:
        save_tokens(r.json())
    except Exception as e:
        st.error(f"Could not parse token response: {e}")
        st.query_params.clear()
        st.stop()

    st.query_params.clear()

def _auth():
    return {**auth_header(st.session_state.oauth["access_token"]), "Accept": "application/json"}

def _auth_v2():
    return {
        **auth_header(st.session_state.oauth["access_token"]),
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

# =========================
# API HELPERS
# =========================
def fetch_all_accounts_v2(limit=2000):
    url = f"{API_V2}/accounts/search"
    offset = 0
    rows = []
    payload = [{"field": "name", "value": "", "criteria": "not_null"}]
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
# LAYOUT: HEADER + NAV
# =========================
handle_callback()

if need_login():
    render_login_page()
    st.stop()

if time.time() > st.session_state.oauth.get("expires_at", 0):
    with st.spinner("Session wird erneuert …"):
        refresh_access_token()

render_header()
sidebar_nav()
st.markdown("---")


# =========================
# STEP 1 — KONTENPLAN
# =========================
if st.session_state.step == 1:
    st.subheader("1) Kontenplan aus bexio importieren")
    st.caption("Nur zum Einlesen der Mapping-Tabelle.")

    if st.button("Kontenplan importieren"):
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
        df["number_str"] = df["number"].astype(str).str.strip()
        df["name_str"]   = df["name"].astype(str)
        is_bank_like = (
            df["number_str"].str.match(r"10\d{2}(-[A-Z])?$", na=False) |
            df["name_str"].str.contains(r"\b(bank|kasse|konto|cash)\b", case=False, regex=True, na=False)
        )
        active_mask = (df["active"] == True)
        banks_df = df[active_mask & is_bank_like].copy()
        if banks_df.empty:
            banks_df = df[active_mask].copy()

        opts = [f"{r['number_str']} · {r['name_str']} (id {int(r['id'])})" for _, r in banks_df.iterrows()]
        choice = st.selectbox("Bankkonto (optional)", ["<keins>"] + opts, index=0)
        if choice != "<keins>":
            st.session_state.selected_bank_number = choice.split(" · ", 1)[0].strip()
            st.caption(
                f"Gewählt: **{st.session_state.selected_bank_number}** – in Schritt 3 wird für **positive Beträge** automatisch **soll**, "
                f"für **negative Beträge** automatisch **haben** mit diesem Konto vorbelegt (falls leer)."
            )

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
    st.caption("Lade eine CSV (oder Excel mit .csv-Endung) – oder starte ohne Datei mit einer leeren Tabelle.")

    bank_file = st.file_uploader(
        "Bankdatei hochladen (CSV/Excel)",
        type=["csv", "xlsx", "xls"]
    )

    # NEW: Skip file upload and go straight to an empty grid
    if st.button("Ohne Datei starten (leere Tabelle)", key="start_empty_table"):
        df_new = pd.DataFrame({
            "csv_row":        [1],
            "buchungsnummer": [""],
            "datum":          [dt_date.today().isoformat()],
            "beschreibung":   [""],
            "betrag":         [0.0],
            "currency":       [DEFAULT_CURRENCY_CODE],
            "exchange_rate":  [1.0],
            "soll":           [""],
            "haben":          [""],
            "mwst_code":      [""],
            "mwst_konto":     [""],
        })

        st.session_state.bulk_df = ensure_schema(df_new)
        st.session_state.pop("bulk_grid", None)
        st.session_state.step = 3
        st.rerun()

    # --- Encoding & Decimal controls (used by the reader) ---
    col_enc1, col_enc2 = st.columns([2, 1])
    encoding_choice = col_enc1.selectbox(
        "Encoding",
        ["utf-8", "utf-8-sig", "latin-1", "cp1252", "utf-16", "utf-16le", "utf-16be"],
        index=0,
    )
    decimal_choice = col_enc2.selectbox("Dezimaltrennzeichen", [".", ","], index=0)

    # --- robust read + preview ---
    if bank_file is not None:
        try:
            st.session_state.bank_csv_df = read_csv_or_excel(bank_file, encoding_choice, decimal_choice)
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

            df_view = df_src.iloc[start_idx:].copy()
            if "csv_row" in df_view.columns:
                df_view = df_view.rename(columns={"csv_row": "csv_row_file"})
            df_view.insert(0, "csv_row", df_view.index + 1)
            df_view = df_view.reset_index(drop=True)

            st.session_state.bank_csv_view_df = df_view

            st.success(f"Verwende {len(df_view)} Datenzeilen ab Zeile {st.session_state.bank_start_row}.")
            st.dataframe(df_view.head(50), use_container_width=True, hide_index=True)

        except Exception as e:
            st.session_state.bank_csv_view_df = None
            st.error(f"CSV konnte nicht gelesen werden: {e}")

    src_for_mapping = st.session_state.get("bank_csv_view_df", None)

    if isinstance(src_for_mapping, pd.DataFrame) and not src_for_mapping.empty:
        st.subheader("Spalten zuordnen (CSV → Bexio-Felder)")

        cols = ["<keine>"] + [c for c in src_for_mapping.columns if c != "csv_row"]

        c1, c2 = st.columns(2)
        st.session_state.bank_map["datum"]  = c1.selectbox("datum", options=cols, index=0)
        st.session_state.bank_map["betrag"] = c2.selectbox("betrag (positiv = Debit / negativ = Kredit)", options=cols, index=0)

        c3, _ = st.columns([2, 1])
        st.session_state.bank_map["beschreibung"] = c3.selectbox("beschreibung (Beschreibung / Text)", options=cols, index=0)

        def pick(key: str) -> pd.Series:
            sel = st.session_state.bank_map.get(key)
            if sel and sel in src_for_mapping.columns:
                return src_for_mapping[sel]
            return pd.Series([""] * len(src_for_mapping), index=src_for_mapping.index, dtype="string")

        def convert_to_grid_and_advance():
            src = src_for_mapping
            df_new = pd.DataFrame({
                "csv_row":        src["csv_row"],
                "buchungsnummer": "",
                "datum":          pick("datum").apply(_parse_date_to_iso),
                "beschreibung":   pick("beschreibung").astype(str),
                "betrag":         pick("betrag").apply(_to_float),
                "currency":       DEFAULT_CURRENCY_CODE,
                "exchange_rate":  1.0,
                "soll":           "",
                "haben":          "",
                "mwst_code":      "",
                "mwst_konto":     "",
            })

            if (df_new["datum"] == "").all():
                df_new["datum"] = dt_date.today().isoformat()

            if st.session_state.selected_bank_number:
                pos_mask = pd.to_numeric(df_new["betrag"], errors="coerce").fillna(0) > 0
                neg_mask = pd.to_numeric(df_new["betrag"], errors="coerce").fillna(0) < 0
                df_new.loc[pos_mask & (df_new["soll"].str.strip() == ""),  "soll"]  = st.session_state.selected_bank_number
                df_new.loc[neg_mask & (df_new["haben"].str.strip() == ""), "haben"] = st.session_state.selected_bank_number

            df_new["betrag"] = pd.to_numeric(df_new["betrag"], errors="coerce").abs()

            st.session_state.bulk_df = ensure_schema(df_new)
            st.session_state.pop("bulk_grid", None)
            st.session_state.step = 3
            st.rerun()

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

    # --- Keyword → Konto rules (persist in session) ---
    if "keyword_rules" not in st.session_state:
        st.session_state.keyword_rules = [
            {"keyword": "Polizei",       "account": "2100", "side": "haben"},
            {"keyword": "Bancomatbezug", "account": "1000", "side": "auto"},
        ]
    if "show_kw_rules" not in st.session_state:
        st.session_state.show_kw_rules = False

    cols_head = st.columns([1, 3, 2])
    if cols_head[0].button("🔎 Schlüsselwörter", key="toggle_kw_rules"):
        st.session_state.show_kw_rules = not st.session_state.show_kw_rules

    if st.session_state.show_kw_rules:
        with st.expander("Schlüsselwörter → Konto-Zuordnung (automatisch ausfüllen)", expanded=True):
            k1, k2, k3, k4 = st.columns([2, 2, 2, 1])
            new_kw   = k1.text_input("Keyword (z.B. Polizei)", key="kw_new_keyword")
            new_kto  = k2.text_input("Konto (Kontonummer oder account_id)", key="kw_new_account")
            new_side = k3.selectbox("Zielspalte", ["haben", "soll", "auto (nach Vorzeichen)"], index=0, key="kw_new_side")

            def _normalize_side(s: str) -> str:
                s = (s or "").lower()
                if s.startswith("auto"):
                    return "auto"
                return "soll" if s == "soll" else "haben"

            if k4.button("➕ Regel hinzufügen", key="kw_add_btn"):
                if new_kw.strip() and new_kto.strip():
                    st.session_state.keyword_rules.append({
                        "keyword": new_kw.strip(),
                        "account": new_kto.strip(),
                        "side": _normalize_side(new_side),
                    })
                    st.session_state.kw_new_keyword = ""
                    st.session_state.kw_new_account = ""

            if st.session_state.keyword_rules:
                st.dataframe(pd.DataFrame(st.session_state.keyword_rules), use_container_width=True, hide_index=True)
            else:
                st.info("Noch keine Regeln erfasst.")

            if st.button("⚙️ Regeln anwenden", type="primary", key="kw_apply_btn"):
                st.session_state.bulk_df = apply_keyword_rules_to_df(
                    st.session_state.bulk_df, st.session_state.keyword_rules
                )
                st.session_state.bulk_df = ensure_schema(st.session_state.bulk_df)
                st.rerun()

    # --- Editable grid (inkl. MWST) ---
    EDIT_COLS = [
        "buchungsnummer", "datum", "beschreibung",
        "betrag", "currency", "exchange_rate",
        "soll", "haben",
        "mwst_code", "mwst_konto",
    ]

    with st.expander("📋 Mehrere Zeilen einfügen (Excel/TSV)", expanded=True):
        paste_text = st.text_area(
            "Füge hier Zeilen ein (Tab-getrennt aus Excel). Erwartet Spalten:\n"
            "datum, beschreibung, betrag, currency(optional), exchange_rate(optional), soll, haben, mwst_code(optional), mwst_konto(optional)",
            height=180,
            key="paste_tsv"
        )

    # ✅ FIXED: indentation + multicurrency detection that won't confuse account numbers with currency/rate
    if st.button("➡️ Eingefügte Zeilen ins Grid laden", key="load_paste_btn"):
        txt = (paste_text or "").strip()
        if not txt:
            st.warning("Kein Text eingefügt.")
        else:
            if "\t" in txt:
                df_p = pd.read_csv(io.StringIO(txt), sep="\t", header=None, dtype=str, keep_default_na=False)
            else:
                df_p = pd.read_csv(
                    io.StringIO(txt),
                    sep=r"\s{2,}",
                    engine="python",
                    header=None,
                    dtype=str,
                    keep_default_na=False
                )

            if df_p.shape[1] < 5:
                st.error(f"Zu wenige Spalten erkannt ({df_p.shape[1]}).")
                st.stop()

            n = df_p.shape[1]

            # defaults: date, desc, amount, soll, haben, (optional) mwst_code, mwst_konto
            currency_col = None
            rate_col = None
            soll_col = 3
            haben_col = 4
            mwst_code_col = 5
            mwst_konto_col = 6

            def _looks_like_ccy(x) -> bool:
                s = str(x).strip().upper()
                if len(s) == 3 and s.isalpha():
                    return True
                if s.isdigit():
                    # currency_id typically small; prevent false positives on account numbers like 1020/2000
                    try:
                        v = int(s)
                        return 1 <= v <= 999
                    except Exception:
                        return False
                return False

            def _looks_like_rate(x) -> bool:
                try:
                    v = float(str(x).strip().replace(",", "."))
                    return 0 < v < 200
                except Exception:
                    return False

            def _looks_like_account(x) -> bool:
                s = str(x).strip().replace("’", "").replace("'", "").replace(" ", "")
                s = s.split("-", 1)[0]
                return s.isdigit() and 1 <= int(s) <= 999999

            # Detect multicurrency safely (7-col multicurrency OR 9-col multicurrency+VAT)
            is_multicurrency = False
            if n >= 7:
                df_probe = df_p.replace("", pd.NA).dropna(how="all")
                if len(df_probe) > 0:
                    ccy = df_probe.iloc[0, 3]
                    rate = df_probe.iloc[0, 4]
                    soll_candidate = df_probe.iloc[0, 5] if n > 5 else ""
                    haben_candidate = df_probe.iloc[0, 6] if n > 6 else ""
                    is_multicurrency = (
                        _looks_like_ccy(ccy)
                        and _looks_like_rate(rate)
                        and _looks_like_account(soll_candidate)
                        and _looks_like_account(haben_candidate)
                    )

            if is_multicurrency:
                currency_col = 3
                rate_col = 4
                soll_col = 5
                haben_col = 6
                mwst_code_col = 7
                mwst_konto_col = 8

            if df_p.shape[1] <= max(soll_col, haben_col):
                st.error(
                    f"Paste-Format passt nicht: soll/haben Spalten fehlen "
                    f"(erkannt: {df_p.shape[1]} Spalten)."
                )
                st.stop()

            df_new = pd.DataFrame({
                "csv_row":        range(1, len(df_p) + 1),
                "buchungsnummer": "",
                "datum":          df_p[0].apply(_parse_date_to_iso),
                "beschreibung":   df_p[1].astype(str),
                "betrag":         df_p[2].apply(_to_float).abs(),
                "currency":       df_p[currency_col].astype(str) if (currency_col is not None and df_p.shape[1] > currency_col) else DEFAULT_CURRENCY_CODE,
                "exchange_rate":  df_p[rate_col].apply(_to_float) if (rate_col is not None and df_p.shape[1] > rate_col) else 1.0,
                "soll":           df_p[soll_col].astype(str),
                "haben":          df_p[haben_col].astype(str),
                "mwst_code":      df_p[mwst_code_col].astype(str) if df_p.shape[1] > mwst_code_col else "",
                "mwst_konto":     df_p[mwst_konto_col].astype(str) if df_p.shape[1] > mwst_konto_col else "",
            })

            st.session_state.bulk_df = ensure_schema(df_new)
            st.session_state.pop("bulk_grid", None)
            st.success(f"{len(df_new)} Zeile(n) ins Grid geladen.")
            st.rerun()

    with st.form("bulk_entries_form", clear_on_submit=False):
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
                "currency":       st.column_config.TextColumn("currency (ISO like CHF/EUR or currency_id)"),
                "exchange_rate":  st.column_config.NumberColumn("exchange rate (currency_factor)", min_value=0.0, step=0.0001, format="%.6f"),
                "soll":           st.column_config.TextColumn("debit (Kontonummer or account_id)"),
                "haben":          st.column_config.TextColumn("credit (Kontonummer or account_id)"),
                "mwst_code":      st.column_config.TextColumn("MWST code (UN81 / VM81 / VB81 / BZM81)"),
                "mwst_konto":     st.column_config.TextColumn("MWST Konto (optional; 3xxx/6xxx if you want to override)"),
            }
        )
        colA, colB = st.columns(2)
        auto_ref   = colA.checkbox("Referenznummer automatisch beziehen (wenn leer)", value=True)
        submitted  = colB.form_submit_button("Buchungen posten 🤖", type="primary")

    # merge edited values back
    st.session_state.bulk_df.loc[:, EDIT_COLS] = edited_view

    # Sanitize editor artifacts
    st.session_state.bulk_df[EDIT_COLS] = (
        st.session_state.bulk_df[EDIT_COLS]
          .replace({None: "", "None": "", "nan": "", "NaN": "", "<NA>": ""})
          .fillna("")
    )
    st.session_state.bulk_df = ensure_schema(st.session_state.bulk_df)

    # ---------- batching controls ----------
    colC, colD = st.columns(2)
    batch_size = int(colC.number_input("Batch-Grösse", min_value=1, max_value=200, value=50, step=1))
    sleep_between_batches = float(colD.number_input("Pause zwischen Batches (Sek.)", min_value=0.0, value=0.0, step=0.1))
    # ---------------------------------------

    # Mapping: VAT code → VAT ledger (fallback if Bexio requires explicit tax_account_id)
    VAT_CODE_TO_LEDGER = {
        "VM81": "1170",
        "UN81": "2200",
        "VB81": "1171",
    }

    # ---- tax-code → tax_id mapper (v3 then v2; scans names/labels; semantic fallback) ----
    def _ensure_tax_code_map():
        if st.session_state.get("tax_code_to_id"):
            return

        def _ingest(items, mp):
            for t in items or []:
                try:
                    tid = int(t.get("id"))
                except Exception:
                    continue

                parts = []
                for k in ("code", "abbreviation", "short_name", "key", "name", "label"):
                    v = t.get(k)
                    if v:
                        parts.append(str(v))
                joined_upper = " ".join(parts).upper()

                for token in ("UN81", "VM81", "VB81", "BZM81"):
                    if token in joined_upper:
                        mp.setdefault(token, tid)

                rate_val = None
                for k in ("rate", "tax_rate", "percent", "value"):
                    if t.get(k) is not None:
                        try:
                            rate_val = float(str(t[k]).replace("%", "").replace(",", "."))
                        except Exception:
                            pass
                        break
                if rate_val is not None and rate_val > 1.0:
                    rate_val = rate_val / 100.0

                def _near_81(x):
                    return (x is not None) and (abs(x - 0.081) < 0.002)

                if _near_81(rate_val):
                    if ("UMSATZ" in joined_upper or " UN " in joined_upper or "UN-" in joined_upper) and "UN81" not in mp:
                        mp.setdefault("UN81", tid)
                    if ("MAT/DL" in joined_upper or " MAT " in joined_upper) and "VM81" not in mp:
                        mp.setdefault("VM81", tid)
                    if ("INV/BA" in joined_upper or " INV " in joined_upper or " BA " in joined_upper) and "VB81" not in mp:
                        mp.setdefault("VB81", tid)

            return mp

        mp = {}

        # Try v3
        try:
            r = requests.get(f"{API_V3}/accounting/taxes", headers=_auth(), timeout=30)
            if r.status_code == 401:
                refresh_access_token()
                r = requests.get(f"{API_V3}/accounting/taxes", headers=_auth(), timeout=30)
            if r.status_code < 400:
                data = r.json()
                _ingest(data if isinstance(data, list) else [], mp)
        except Exception:
            pass

        # Fallback to v2
        if not mp:
            try:
                r2 = requests.get(f"{API_V2}/taxes", headers=_auth_v2(), timeout=30)
                if r2.status_code == 401:
                    refresh_access_token()
                    r2 = requests.get(f"{API_V2}/taxes", headers=_auth_v2(), timeout=30)
                if r2.status_code < 400:
                    data2 = r2.json()
                    _ingest(data2 if isinstance(data2, list) else [], mp)
            except Exception:
                pass

        st.session_state.tax_code_to_id = mp

    def _tax_id_from_code(code_str: str) -> int | None:
        code = (code_str or "").upper().strip()
        if not code:
            return None

        env_key = f"BEXIO_TAX_ID_{code}"
        env_val = os.getenv(env_key)
        if env_val:
            try:
                return int(env_val)
            except ValueError:
                pass

        _ensure_tax_code_map()
        return st.session_state.get("tax_code_to_id", {}).get(code)

    # === DEBUG BUTTON: raw JSON for today's manual entries ===
    if st.button("Debug: raw JSON for today's entries- Peace"):
        try:
            r = requests.get(
                MANUAL_ENTRIES_V3,
                headers=_auth(),
                params={"limit": 2000},
                timeout=30,
            )
            st.write("status manual_entries:", r.status_code)
            data = r.json()

            if isinstance(data, list) and data:
                today_str = dt_date.today().isoformat()
                st.write("Heute:", today_str)

                todays_entries = [e for e in data if e.get("date") == today_str]

                if not todays_entries:
                    st.write("Keine Buchungen mit heutigem Datum gefunden.")
                else:
                    st.write(f"{len(todays_entries)} Buchung(en) mit heutigem Datum gefunden.")
                    for e in todays_entries:
                        st.json(e)
            else:
                st.write("Antwort:", data)
        except Exception as e:
            st.write("manual_entries raw-json error:", str(e))

    if submitted:
        rows = st.session_state.bulk_df.copy()

        st.write("DEBUG: bulk_df rows:", len(st.session_state.bulk_df))
        st.write("DEBUG: rows copied for import:", len(rows))
        st.dataframe(rows[EDIT_COLS].head(30), use_container_width=True, hide_index=True)

        if rows.empty:
            st.warning("Keine Zeilen im Gitter.")
        else:
            results = []
            idxs = list(rows.index)

            def _s(x):
                try:
                    if x is None or pd.isna(x):
                        return ""
                except Exception:
                    pass
                return str(x)

            def _f(x):
                try:
                    return float(x)
                except Exception:
                    return 0.0

            for start in range(0, len(idxs), batch_size):
                chunk_idxs = idxs[start:start + batch_size]

                for idx in chunk_idxs:
                    row = rows.loc[idx]
                    try:
                        # Skip empty editor rows
                        if (_s(row.get("datum")) == "" and
                            _f(row.get("betrag")) == 0 and
                            _s(row.get("soll")) == "" and
                            _s(row.get("haben")) == ""):
                            continue

                        # date
                        date_val = row.get("datum")
                        date_iso = date_val.isoformat() if isinstance(date_val, dt_date) else _parse_date_to_iso(_s(date_val))
                        if not date_iso:
                            raise ValueError(f"Ungültiges Datum in Editor-Zeile {idx+1}")

                        # amount (absolute)
                        amount = abs(_f(row.get("betrag")))
                        if amount == 0:
                            raise ValueError(f"Betrag darf nicht 0 sein (Editor-Zeile {idx+1}).")

                        # accounts
                        debit_id  = resolve_account_id_from_number_or_id(row.get("soll"))
                        credit_id = resolve_account_id_from_number_or_id(row.get("haben"))
                        if not debit_id or not credit_id:
                            raise ValueError(f"Konto unbekannt (Editor-Zeile {idx+1}): Kontonummern prüfen / Kontenplan importieren.")

                        # reference number
                        ref_nr = _s(row.get("buchungsnummer"))
                        if auto_ref and not ref_nr:
                            rr = requests.get(NEXT_REF_V3, headers=_auth(), timeout=15)
                            rr.raise_for_status()
                            ref_nr = (rr.json() or {}).get("next_ref_nr") or ""

                        desc      = _s(row.get("beschreibung"))
                        code_raw  = _s(row.get("mwst_code")).upper().strip()

                        tax_id = None
                        if code_raw:
                            tax_id = _tax_id_from_code(code_raw)
                            if not tax_id:
                                raise ValueError(
                                    f"MWST-Code '{code_raw}' konnte nicht auf tax_id gemappt werden. "
                                    f"Bitte prüfe im bexio-Mandanten, ob der Satz aktiv ist und "
                                    f"‘{code_raw}’ im Namen/Kürzel enthält oder setze BEXIO_TAX_ID_{code_raw} in der .env."
                                )

                        currency_raw = _s(row.get("currency")).upper().strip()
                        rate_raw = row.get("exchange_rate")
                        rate = _f(rate_raw)

                        if not currency_raw:
                            currency_raw = DEFAULT_CURRENCY_CODE

                        currency_id = _currency_id_from_input(currency_raw)
                        if not currency_id:
                            raise ValueError(
                                f"Unbekannte Währung '{currency_raw}' (Editor-Zeile {idx+1}). "
                                f"Nutze ISO (CHF/EUR/USD) oder trage currency_id ein. Optional: setze BEXIO_CURRENCY_ID_{currency_raw} in .env."
                            )

                        if currency_raw == DEFAULT_CURRENCY_CODE or int(currency_id) == int(DEFAULT_CURRENCY_ID):
                            currency_factor = 1.0
                        else:
                            if rate <= 0:
                                raise ValueError(
                                    f"Exchange rate fehlt/ungültig für {currency_raw} (Editor-Zeile {idx+1}). "
                                    f"Bitte positiven Kurs eintragen (z.B. 0.95)."
                                )
                            currency_factor = float(rate)

                        base_entry = {
                            "debit_account_id": int(debit_id),
                            "credit_account_id": int(credit_id),
                            "amount": float(amount),
                            "description": desc,
                            "currency_id": int(currency_id),
                            "currency_factor": float(currency_factor),
                        }

                        if tax_id:
                            base_entry["tax_id"] = int(tax_id)
                            base_entry["tax_account_id"] = int(debit_id)

                        def _post_with_entry(entry_obj):
                            payload_local = {
                                "type": "manual_single_entry",
                                "date": date_iso,
                                "entries": [entry_obj],
                            }
                            if ref_nr:
                                payload_local["reference_nr"] = ref_nr
                            return post_with_backoff(
                                MANUAL_ENTRIES_V3,
                                headers={**_auth(), "Content-Type": "application/json"},
                                payload=payload_local,
                            )

                        ok, resp = _post_with_entry(base_entry)

                        if ok:
                            try:
                                rid = resp.json().get("id")
                            except Exception:
                                rid = None
                            results.append({
                                "row": idx + 1, "csv_row": row.get("csv_row", ""),
                                "status": "OK", "id": rid, "reference_nr": ref_nr
                            })
                        else:
                            results.append({
                                "row": idx + 1, "csv_row": row.get("csv_row", ""),
                                "status": "ERROR", "error": resp
                            })

                    except requests.HTTPError as e:
                        try:
                            err_txt = e.response.text
                        except Exception:
                            err_txt = str(e)
                        results.append({
                            "row": idx + 1, "csv_row": row.get("csv_row", ""),
                            "status": f"HTTP {e.response.status_code}", "error": err_txt
                        })
                    except Exception as e:
                        results.append({
                            "row": idx + 1, "csv_row": row.get("csv_row", ""),
                            "status": "ERROR", "error": str(e)
                        })

                if sleep_between_batches and (start + batch_size) < len(idxs):
                    time.sleep(sleep_between_batches)

            if not results:
                st.info("Keine gültigen Zeilen zum Posten gefunden.")
            else:
                st.success(f"Fertig. {sum(1 for r in results if r.get('status')=='OK')} Buchung(en) erfolgreich gepostet.")
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)


