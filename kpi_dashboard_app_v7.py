
import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

st.set_page_config(page_title="A‑Team KPI — Per‑User Hourly (Protected)", page_icon="🔒", layout="wide", initial_sidebar_state="expanded")

# ---------------- Password gate ----------------
def _get_app_password():
    try:
        # Prefer Streamlit Secrets: add in App settings → Advanced → Secrets
        if "auth" in st.secrets and "password" in st.secrets["auth"]:
            return st.secrets["auth"]["password"]
    except Exception:
        pass
    # Fallback to environment variable APP_PASSWORD
    return os.getenv("APP_PASSWORD", None)

def password_gate():
    pw = _get_app_password()
    if not pw:
        st.error("No dashboard password is configured. Set **auth.password** in Streamlit Secrets or **APP_PASSWORD** environment variable.")
        st.stop()

    if st.session_state.get("authenticated", False):
        return True

    st.markdown("### 🔒 Secure login")
    with st.form("login", clear_on_submit=False):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Unlock")
    if submitted:
        if entered == pw:
            st.session_state["authenticated"] = True
            st.success("Unlocked")
            return True
        else:
            st.error("Incorrect password")
    st.stop()

password_gate()

# ---------------- Helpers ----------------
@st.cache_data
def load_table(file):
    if file is None:
        return None
    name = getattr(file, "name", "").lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        file.seek(0)
        try:
            return pd.read_excel(file, engine="openpyxl")
        except Exception:
            file.seek(0)
            return pd.read_excel(file)
    # CSV attempts
    for kwargs in [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "latin-1"},
        {"sep": ";", "encoding": "latin-1"},
    ]:
        try:
            file.seek(0)
            return pd.read_csv(file, **kwargs)
        except Exception:
            continue
    file.seek(0)
    return pd.read_csv(file)

def parse_dt(date_series, time_series=None):
    if time_series is None:
        return pd.to_datetime(date_series, errors="coerce", dayfirst=True)
    s = date_series.astype(str).str.strip() + " " + time_series.astype(str).str.strip()
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def norm_email_or_name(x):
    if pd.isna(x): return None
    s = str(x).strip()
    # emails -> lowercase
    if "@" in s:
        return s.lower()
    # names -> title case
    return " ".join(w.capitalize() for w in s.split() if w)

def find_col(cols, targets, contains_any=None):
    low = [c.lower() for c in cols]
    for t in targets or []:
        t_low = t.lower()
        if t_low in low:
            return cols[low.index(t_low)]
    if contains_any:
        for i, c in enumerate(low):
            if any(tok.lower() in c for tok in contains_any):
                return cols[i]
    return None

def clean_token(s):
    return re.sub(r"[^a-z]", "", s.lower())

def email_candidates(email):
    if not isinstance(email, str): return set()
    local = email.split("@")[0].lower()
    parts = re.split(r"[._\\-+]", local)
    parts = [p for p in parts if p]
    keys = set()
    if parts:
        first = parts[0]
        last = parts[-1] if len(parts) > 1 else ""
        if last:
            keys.add(clean_token(first[:1] + last))
        keys.add(clean_token("".join(parts)))
        keys.add(clean_token(local.replace(".", "").replace("_","").replace("-","")))
        initials = "".join(p[:1] for p in parts)
        keys.add(clean_token(initials))
    else:
        keys.add(clean_token(local))
    return keys

def name_candidates(name):
    if not isinstance(name, str): return set()
    t = re.sub(r"\\s+", " ", name.strip().lower())
    parts = [p for p in re.split(r"[ \\-']", t) if p]
    keys = set()
    if parts:
        first = parts[0]
        last = parts[-1] if len(parts) > 1 else ""
        if last:
            keys.add(clean_token(first[:1] + last))
        keys.add(clean_token("".join(parts)))
        initials = "".join(p[:1] for p in parts)
        keys.add(clean_token(initials))
    else:
        keys.add(clean_token(t))
    return keys

def build_email_matcher(klinik_emails):
    key_to_email = {}
    collisions = set()
    for em in sorted(set([e for e in klinik_emails if isinstance(e, str) and "@" in e])):
        for key in email_candidates(em):
            if key in key_to_email and key_to_email[key] != em:
                collisions.add(key)
            else:
                key_to_email[key] = em
    for k in collisions:
        key_to_email.pop(k, None)
    return key_to_email, collisions

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Controls")
    today = date.today()
    default_start = today - timedelta(days=today.weekday())
    default_end = default_start + timedelta(days=4)
    start = st.date_input("Start date", value=default_start, key="start_date")
    end = st.date_input("End date", value=default_end, key="end_date")
    if end < start:
        st.error("End date must be on or after start date"); st.stop()

    st.subheader("Upload files")
    case_file = st.file_uploader("Klinik Case Counts (CSV/XLSX)", type=["csv","xlsx","xls"], key="case")
    call_file = st.file_uploader("Telephone Calls Export (CSV/XLSX)", type=["csv","xlsx","xls"], key="calls")
    st.caption("We will auto-map Klinik emails to Calls users.")

st.title("🔒 Per‑User Hourly (Auto + Email Match)")
st.caption("Password-protected. Matches **Klinik emails** to **Calls user names** so both sources aggregate under the same person.")

# ---------------- Load ----------------
case_df = load_table(case_file) if case_file else None
call_df = load_table(call_file) if call_file else None

if case_df is None or call_df is None:
    st.info("Upload both **Klinik** and **Calls** files to continue."); st.stop()

# Auto-map Klinik columns
case_cols = case_df.columns
case_staff_col = find_col(case_cols, ["last_archived_by"], contains_any=["archived by","archived_by","staff","user"])
case_date_col  = find_col(case_cols, ["last_archived_date"], contains_any=["archived date","archived_date","date"])
case_time_col  = find_col(case_cols, ["last_archived_time"], contains_any=["archived time","archived_time","time"])
case_unit_col  = find_col(case_cols, [], contains_any=["unit","closed in","closed_in","unit_closed","unit closed"])

missing_klinik = [n for n,v in {"Staff": case_staff_col, "Date": case_date_col, "Time": case_time_col, "Unit": case_unit_col}.items() if v is None]
if missing_klinik:
    st.error("Klinik file: couldn't auto-detect: " + ", ".join(missing_klinik)); st.stop()

# Auto-map Calls columns
call_cols = call_df.columns
call_user_col    = find_col(call_cols, ["User Name"], contains_any=["user name","username","agent"])
call_caller_col  = find_col(call_cols, ["Caller Name"], contains_any=["caller name","caller"])
call_outcome_col = find_col(call_cols, ["Outcome"], contains_any=["outcome","result","status"])
call_direction_col = find_col(call_cols, ["Direction"], contains_any=["direction","inbound","outbound"])
call_start_col   = find_col(call_cols, ["Start Time"], contains_any=["start time","started","datetime","date"])

missing_calls = [n for n,v in {"User Name": call_user_col, "Caller Name": call_caller_col, "Outcome": call_outcome_col, "Direction": call_direction_col, "Start Time": call_start_col}.items() if v is None]
if missing_calls:
    st.error("Calls file: couldn't auto-detect: " + ", ".join(missing_calls)); st.stop()

# ---------------- Build events ----------------
# Klinik
case_dt = parse_dt(case_df[case_date_col], case_df[case_time_col])
case_staff = case_df[case_staff_col].map(norm_email_or_name)
case_units = case_df[case_unit_col].astype(str).replace(["nan","None"], np.nan).fillna("Unknown")
klinik_ev = pd.DataFrame({"when": case_dt, "who_email": case_staff, "kind": "Unit Finished", "unit": case_units})

# Calls
dt = parse_dt(call_df[call_start_col], None)
dir_lower = call_df[call_direction_col].astype(str).str.lower()
out_lower = call_df[call_outcome_col].astype(str).str.lower()
inbound_mask = dir_lower.str.contains("inbound")
answered_mask = out_lower.str.contains("answer|connect|complete|handled|finished|resolved")
who_calls = np.where(inbound_mask, call_df[call_user_col].astype(str), call_df[call_caller_col].astype(str))
who_calls = pd.Series(who_calls)
calls_ev = pd.DataFrame({"when": dt, "who_calls_name": who_calls, "kind": "Call Answer"})
try:
    calls_ev = calls_ev[answered_mask.values]
except Exception:
    pass

# Email ↔ Name Matching
email_series = klinik_ev["who_email"].dropna()
key_to_email, collisions = build_email_matcher(email_series)

def best_email_for_call_name(name):
    cand = name_candidates(name)
    for k in cand:
        if k in key_to_email:
            return key_to_email[k]
    return None

calls_ev["who_email"] = calls_ev["who_calls_name"].apply(best_email_for_call_name)
klinik_ev["person"] = klinik_ev["who_email"]
calls_ev["person"] = calls_ev["who_email"].where(calls_ev["who_email"].notna(), calls_ev["who_calls_name"].apply(norm_email_or_name))

# Combine
events = [klinik_ev[["when","person","kind","unit"]], calls_ev[["when","person","kind"]].assign(unit=np.nan)]
df = pd.concat(events, ignore_index=True).dropna(subset=["when","person"])

# Filter date/time
start_dt = datetime.combine(start, datetime.min.time())
end_dt   = datetime.combine(end, datetime.max.time())
df = df[(df["when"] >= pd.Timestamp(start_dt)) & (df["when"] <= pd.Timestamp(end_dt))].copy()

# Clamp to 08:00–18:30
df = df[((df["when"].dt.hour > 8) | ((df["when"].dt.hour == 8) & (df["when"].dt.minute >= 0))) &
        ((df["when"].dt.hour < 18) | ((df["when"].dt.hour == 18) & (df["when"].dt.minute <= 30)))]

if df.empty:
    st.warning("No events for the selected date range/time window."); st.stop()

# Sidebar user selection uses unified 'person'
with st.sidebar:
    users = sorted(df["person"].dropna().unique().tolist())
    selected_user = st.selectbox("Select user (email preferred)", users)

if not selected_user:
    st.info("Choose a user to continue."); st.stop()

udf = df[df["person"] == selected_user].copy()
if udf.empty:
    st.warning("No events for this user in the selected range."); st.stop()

udf["day"] = udf["when"].dt.strftime("%a %d %b")
udf["hour"] = udf["when"].dt.hour

ordered_days = sorted(udf["day"].unique().tolist(), key=lambda s: datetime.strptime(s, "%a %d %b"))
hours = list(range(8,19))  # 8..18 inclusive

grp = udf.groupby(["day","hour","kind"]).agg(
    count=("kind","size"),
    units=("unit", lambda x: "; ".join(pd.Series(x).dropna().astype(str).value_counts().head(3).index) if "unit" in udf.columns else None)
).reset_index()

rows = []
for d in ordered_days:
    for h in hours:
        row = {"Day": d, "Hour": f"{h:02d}:00"}
        for kind in ["Call Answer","Unit Finished"]:
            c = grp[(grp["day"]==d) & (grp["hour"]==h) & (grp["kind"]==kind)]
            row[kind] = int(c["count"].iloc[0]) if not c.empty else 0
            if kind=="Unit Finished":
                row["Top Units"] = c["units"].iloc[0] if (not c.empty and "units" in c.columns) else ""
        rows.append(row)

grid = pd.DataFrame(rows)

st.markdown(f"### {selected_user} — Hourly results (08:00–18:30)")
st.dataframe(grid, use_container_width=True, height=420)

st.markdown("#### Visual — per hour, by category (stacked)")
for d in ordered_days:
    sub = grid[grid["Day"]==d].copy()
    fig = go.Figure()
    fig.add_bar(x=sub["Hour"], y=sub["Call Answer"], name="Call Answer")
    fig.add_bar(x=sub["Hour"], y=sub["Unit Finished"], name="Unit Finished")
    fig.update_layout(barmode="stack", title=f"{d}", xaxis_title="Hour", yaxis_title="Count", legend_title="Category")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Diagnostics (auto-match summary)", expanded=False):
    st.write("Collision keys (ignored due to ambiguity):", sorted(list(collisions)))
    if key_to_email:
        sample_map = pd.DataFrame([(k, key_to_email[k]) for k in sorted(key_to_email.keys())[:50]], columns=["Key","Email"])
        st.write("Sample key→email map:")
        st.dataframe(sample_map, use_container_width=True)

csv = grid.to_csv(index=False).encode("utf-8")
st.download_button("Download this user's hourly grid (CSV)", data=csv, file_name=f"{str(selected_user).replace(' ','_')}_hourly_grid.csv", mime="text/csv")
