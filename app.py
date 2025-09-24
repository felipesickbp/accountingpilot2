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
API_V2 = "https://api.bexio.com/2.0"

SCOPES = "openid profile email offline_access company_profile"

st.set_page_config(page_title="bexio Bulk Manual Entries (v3)", page_icon="📘", layout="wide")

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
REQUIRED_COLS = ["csv_row", "buchungsnummer", "datum", "betrag", "soll", "haben", "beschreibung"]

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
        df["csv_row"] = pd.Series([pd.NA]*len(df), dtype="Int64")
    df["betrag"] = pd.to_numeric(df["betrag"], errors="coerce").astype(float)
    for c in ["buchungsnummer", "datum", "soll", "haben", "beschreibung"]:
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
    if x is None:
        return 0.0
    s = str(x).strip().replace("’","").replace("'","")
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
    # - drop everything after a dash (e.g. "1020-A" -> "1020")
    # - remove spaces / thousands-separators ' and ’
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
    raw = uploaded_file.getvalue()
    # Excel?
    if len(raw) >= 4 and raw[:4] == b"PK\x03\x04":
        df = pd.read_excel(io.BytesIO(raw), dtype=str, keep_default_na=False)
        df.columns = _make_unique_columns(df.columns, prefix="col")
        if "csv_row" in df.columns:
            df = df.rename(columns={"csv_row": "csv_row_file"})
        return df
    encodings  = [encoding_preference, "utf-8-sig", "cp1252", "latin-1", "utf-16", "utf-16le", "utf-16be"]
    delimiters = [None, ";", ",", "\t", "|"]
    errs = []
    for enc in encodings:
        for sep in delimiters:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python",
                                 encoding=enc, decimal=decimal, dtype=str,
                                 keep_default_na=False, header=0)
                if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
                    df.columns = _make_unique_columns(df.columns, prefix="col")
                    if "csv_row" in df.columns:
                        df = df.rename(columns={"csv_row": "csv_row_file"})
                    return df
            except Exception as e:
                errs.append(f"h=0 {enc}/{repr(sep)} → {e}")
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python",
                                 encoding=enc, decimal=decimal, dtype=str,
                                 keep_default_na=False, header=None)
                if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
                    df.columns = _make_unique_columns([f"col_{i+1}" for i in range(df.shape[1])], prefix="col")
                    if "csv_row" in df.columns:
                        df = df.rename(columns={"csv_row": "csv_row_file"})
                    return df
            except Exception as e:
                errs.append(f"h=None {enc}/{repr(sep)} → {e}")
    raise ValueError("CSV konnte nicht gelesen werden: " + "; ".join(errs[:6]) + (" …" if len(errs) > 6 else ""))

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
    return {**auth_header(st.session_state.oauth["access_token"]),
            "Accept": "application/json",
            "Content-Type": "application/json"}

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
    st.caption("Lade eine CSV (oder Excel mit .csv-Endung). Danach Startzeile und Spalten zuordnen.")

    bank_file = st.file_uploader("Bank-CSV hochladen", type=["csv"])
    col_enc1, col_enc2 = st.columns([2,1])
    encoding = col_enc1.selectbox("Encoding", ["utf-8", "latin-1", "utf-16"], index=0)
    decimal  = col_enc2.selectbox("Dezimaltrennzeichen", [".", ","], index=0)

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
        c3, _ = st.columns([2,1])
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
                "soll":           "",
                "haben":          "",
            })

            if (df_new["datum"] == "").all():
                df_new["datum"] = dt_date.today().isoformat()

            if st.session_state.selected_bank_number:
                pos_mask = pd.to_numeric(df_new["betrag"], errors="coerce").fillna(0) > 0
                neg_mask = pd.to_numeric(df_new["betrag"], errors="coerce").fillna(0) < 0
                df_new.loc[pos_mask & (df_new["soll"].str.strip() == ""),  "soll"]  = st.session_state.selected_bank_number
                df_new.loc[neg_mask & (df_new["haben"].str.strip() == ""), "haben"] = st.session_state.selected_bank_number

            # amount positive
            df_new["betrag"] = pd.to_numeric(df_new["betrag"], errors="coerce").abs()

            st.session_state.bulk_df = ensure_schema(df_new)
            st.session_state.step = 3
            st.rerun()

        if st.button("Weiter → 3) Kontrolle & Import", type="primary"):
            convert_to_grid_and_advance()
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

    EDIT_COLS = ["buchungsnummer", "datum", "beschreibung", "betrag", "soll", "haben"]

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
                "soll":           st.column_config.TextColumn("debit (Kontonummer or account_id)"),
                "haben":          st.column_config.TextColumn("credit (Kontonummer or account_id)"),
            }
        )
        colA, colB = st.columns(2)
        auto_ref   = colA.checkbox("Referenznummer automatisch beziehen (wenn leer)", value=True)
        submitted  = colB.form_submit_button("Buchungen posten", type="primary")

    # merge edited values back into full DF, preserve csv_row
    st.session_state.bulk_df.loc[:, EDIT_COLS] = edited_view
    st.session_state.bulk_df = ensure_schema(st.session_state.bulk_df)

    if submitted:
        rows = st.session_state.bulk_df.copy()
        if rows.empty:
            st.warning("Keine Zeilen im Gitter.")
        else:
            results = []
            for idx, row in rows.iterrows():
                try:
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

                    debit_id  = resolve_account_id_from_number_or_id(row.get("soll"))
                    credit_id = resolve_account_id_from_number_or_id(row.get("haben"))
                    if not debit_id or not credit_id:
                        raise ValueError(f"Konto unbekannt (Editor-Zeile {idx+1}): Kontonummern prüfen / Kontenplan importieren.")

                    ref_nr = _s(row.get("buchungsnummer"))
                    if auto_ref and not ref_nr:
                        rr = requests.get(NEXT_REF_V3, headers=_auth(), timeout=15)
                        rr.raise_for_status()
                        ref_nr = (rr.json() or {}).get("next_ref_nr") or ""

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
                    results.append({"row": idx + 1, "csv_row": row.get("csv_row",""), "status": "OK",
                                    "id": r.json().get("id"), "reference_nr": ref_nr})
                except requests.HTTPError as e:
                    try:
                        err_txt = e.response.text
                    except Exception:
                        err_txt = str(e)
                    results.append({"row": idx + 1, "csv_row": row.get("csv_row",""),
                                    "status": f"HTTP {e.response.status_code}", "error": err_txt})
                except Exception as e:
                    results.append({"row": idx + 1, "csv_row": row.get("csv_row",""), "status": "ERROR", "error": str(e)})

            if not results:
                st.info("Keine gültigen Zeilen zum Posten gefunden.")
            else:
                st.success(f"Fertig. {sum(1 for r in results if r.get('status')=='OK')} Buchung(en) erfolgreich gepostet.")
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)


