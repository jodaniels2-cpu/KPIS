import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

st.set_page_config(page_title="Modality Lewisham — KPI Dashboard (v9d2: robust pickers)", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
:root { --ml-primary:#005eb8; --ml-accent:#00a3a3; --ml-muted:#f2f7ff; }
.ml-header{background:var(--ml-muted);border:1px solid #e6eefc;padding:14px 18px;border-radius:16px;display:flex;align-items:center;gap:14px;margin-bottom:10px;}
.ml-pill{background:var(--ml-primary);color:#fff;font-weight:700;padding:4px 10px;border-radius:999px;font-size:12px;letter-spacing:.3px;}
.ml-title{margin:0;font-weight:800;font-size:22px;color:#0b2e59;}
.ml-sub{margin:0;color:#345;font-size:13px;}
.stDataFrame{border-radius:12px;overflow:hidden;border:1px solid #eef3ff;}
.stButton > button, .stDownloadButton > button { border-radius:10px; }
</style>
""", unsafe_allow_html=True,
)

st.markdown(
    """
<div class=\"ml-header\">
  <span class=\"ml-pill\">Modality Lewisham</span>
  <div>
    <p class=\"ml-title\">A‑Team KPI Dashboard</p>
    <p class=\"ml-sub\">Three independent sections with their own staff pickers: Klinik • Docman • Calls</p>
  </div>
</div>
""", unsafe_allow_html=True,
)

def _get_app_password():
    try:
        if "auth" in st.secrets and "password" in st.secrets["auth"]:
            return st.secrets["auth"]["password"]
    except Exception:
        pass
    return os.getenv("APP_PASSWORD", None)

def password_gate():
    pw = _get_app_password()
    if not pw:
        st.error("No dashboard password is configured. Set **auth.password** in Secrets or **APP_PASSWORD** env var.")
        st.stop()
    if st.session_state.get("authenticated", False):
        return True
    with st.form("login", clear_on_submit=False):
        st.subheader("🔒 Enter password to continue")
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
            file.seek(0); return pd.read_excel(file)
    for kwargs in [{"sep":",","encoding":"utf-8"},{"sep":";","encoding":"utf-8"},{"sep":",","encoding":"latin-1"},{"sep":";","encoding":"latin-1"}]:
        try:
            file.seek(0); return pd.read_csv(file, **kwargs)
        except Exception:
            continue
    file.seek(0); return pd.read_csv(file)

def parse_dt(date_series, time_series=None):
    if time_series is None:
        return pd.to_datetime(date_series, errors="coerce", dayfirst=True, infer_datetime_format=True)
    ds = pd.Series(date_series).astype(str).str.strip()
    ts = pd.Series(time_series).astype(str).str.strip()
    s = (ds + " " + ts).str.replace(r"\s+", " ", regex=True)
    out = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
    if out.notna().mean() < 0.5:
        alt = pd.to_datetime(date_series, errors="coerce", dayfirst=True, infer_datetime_format=True)
        out = out.where(out.notna(), alt)
    return out

def find_col(cols, targets=None, contains_any=None):
    targets = targets or []
    low=[c.lower() for c in cols]
    for t in targets:
        if t.lower() in low: return cols[low.index(t.lower())]
    if contains_any:
        for i,c in enumerate(low):
            if any(tok.lower() in c for tok in contains_any): return cols[i]
    return None

def normalise_name_or_email(x):
    if pd.isna(x): return None
    s = str(x).strip()
    return s.lower() if "@" in s else s

with st.sidebar:
    st.header("Global filters")
    today=date.today()
    default_start=today - timedelta(days=today.weekday())
    default_end=default_start + timedelta(days=4)
    start=st.date_input("Start date", value=default_start)
    end=st.date_input("End date", value=default_end)
    if end<start: st.error("End date must be on or after start date"); st.stop()
    clamp = st.checkbox("Limit to 08:00–18:30", value=True)
    only_answered = st.checkbox("Calls: only include answered/connected", value=True)
    st.subheader("Upload files (all three required)")
    case_file=st.file_uploader("Klinik Case Counts (CSV/XLSX)", type=["csv","xlsx","xls"])
    doc_file=st.file_uploader("Docman Tasks (CSV/XLSX)", type=["csv","xlsx","xls"])
    call_file=st.file_uploader("Telephone Calls Export (CSV/XLSX)", type=["csv","xlsx","xls"])

case_df=load_table(case_file) if case_file else None
doc_df=load_table(doc_file) if doc_file else None
call_df=load_table(call_file) if call_file else None
if any(x is None for x in (case_df, doc_df, call_df)):
    st.info("Upload **all three** files to continue."); st.stop()

# Mapping
case_staff_col = find_col(case_df.columns, ["last_archived_by"], contains_any=["archived by","staff","user"])
case_date_col  = find_col(case_df.columns, ["last_archived_date"], contains_any=["archived date","date"])
case_time_col  = find_col(case_df.columns, ["last_archived_time"], contains_any=["archived time","time"])
case_unit_col  = find_col(case_df.columns, ["unit_closed_in","unit"], contains_any=["unit","closed in"])
if any(v is None for v in [case_staff_col, case_date_col, case_unit_col]):
    st.error("Klinik file: couldn't auto-detect required columns."); st.stop()

doc_user_col = find_col(doc_df.columns, ["User","Completed User"], contains_any=["user","completed"])
doc_dt_col   = find_col(doc_df.columns, ["Date and Time of Event"], contains_any=["date and time of event","completed","datetime","date","time"])
if doc_dt_col is None:
    for c in doc_df.columns:
        try:
            test = pd.to_datetime(doc_df[c], errors="coerce")
            if test.notna().mean()>0.7:
                doc_dt_col=c; break
        except Exception:
            pass
if any(v is None for v in [doc_user_col, doc_dt_col]):
    st.error("Docman file: couldn't auto-detect required columns."); st.stop()

call_user_answered_col = find_col(call_df.columns, ["User Name","Answered","Answered By","Agent"], contains_any=["user name","answered","agent","owner","user"])
call_caller_col        = find_col(call_df.columns, ["Caller Name","Caller"], contains_any=["caller","callback"])
call_type_col          = find_col(call_df.columns, ["Call Type","Type"], contains_any=["type","call type","group","callback","cb"])
call_outcome_col       = find_col(call_df.columns, ["Outcome"], contains_any=["outcome","result","status","disposition"])
call_start_col         = find_col(call_df.columns, ["Start Time","Start","StartDateTime","Start Datetime","Call Start Time","Call Started"], contains_any=["start time","start","begin"])
if any(v is None for v in [call_user_answered_col, call_caller_col, call_type_col, call_start_col]):
    st.error("Calls file: couldn't auto-detect required columns."); st.stop()

# Build datasets
start_dt = datetime.combine(start, datetime.min.time())
end_dt = datetime.combine(end, datetime.max.time())

# Klinik
case_dt_all = parse_dt(case_df[case_date_col], case_df[case_time_col] if case_time_col in case_df.columns else None) if case_time_col else parse_dt(case_df[case_date_col])
klinik_all = pd.DataFrame({
    "when": case_dt_all,
    "person": case_df[case_staff_col].map(normalise_name_or_email),
    "unit": case_df[case_unit_col].astype(str).replace(["nan","None",""], np.nan).fillna("Unknown")
}).dropna(subset=["when","person"])
klinik_all = klinik_all[(klinik_all["when"]>=pd.Timestamp(start_dt)) & (klinik_all["when"]<=pd.Timestamp(end_dt))]
klinik = klinik_all.copy()
if clamp:
    klinik = klinik[((klinik["when"].dt.hour > 8) | ((klinik["when"].dt.hour==8) & (klinik["when"].dt.minute>=0))) &
                    ((klinik["when"].dt.hour < 18) | ((klinik["when"].dt.hour==18) & (klinik["when"].dt.minute<=30)))]

# Docman
doc_dt_all = parse_dt(doc_df[doc_dt_col], None)
docman_all = pd.DataFrame({
    "when": doc_dt_all,
    "person": doc_df[doc_user_col].map(normalise_name_or_email)
}).dropna(subset=["when","person"])
docman_all = docman_all[(docman_all["when"]>=pd.Timestamp(start_dt)) & (docman_all["when"]<=pd.Timestamp(end_dt))]
docman = docman_all.copy()
if clamp:
    docman = docman[((docman["when"].dt.hour > 8) | ((docman["when"].dt.hour==8) & (docman["when"].dt.minute>=0))) &
                    ((docman["when"].dt.hour < 18) | ((docman["when"].dt.hour==18) & (docman["when"].dt.minute<=30)))]

# Calls
call_dt_all = parse_dt(call_df[call_start_col], None)
type_lower = call_df[call_type_col].astype(str).str.lower()
is_callback = type_lower.str.contains("callback|call back|cb", na=False)
who_all = np.where(is_callback, call_df[call_caller_col].astype(str), call_df[call_user_answered_col].astype(str))

calls_all = pd.DataFrame({
    "when": call_dt_all,
    "person": pd.Series(who_all).map(normalise_name_or_email),
    "kind": np.where(is_callback, "Callback","Group"),
    "outcome": call_df[call_outcome_col].astype(str).str.lower() if call_outcome_col else ""
}).dropna(subset=["when","person"])
calls_all = calls_all[(calls_all["when"]>=pd.Timestamp(start_dt)) & (calls_all["when"]<=pd.Timestamp(end_dt))]

calls = calls_all.copy()
if only_answered and call_outcome_col is not None:
    answered_mask = calls["outcome"].str.contains("answer|answered|connect|connected|complete|completed|handled|finished|resolved|success|ok", na=False)
    calls = calls[answered_mask]
if clamp:
    calls = calls[((calls["when"].dt.hour > 8) | ((calls["when"].dt.hour==8) & (calls["when"].dt.minute>=0))) &
                  ((calls["when"].dt.hour < 18) | ((calls["when"].dt.hour==18) & (calls["when"].dt.minute<=30)))]

# UI
st.markdown("## Klinik")
k_users = sorted(klinik_all["person"].dropna().unique().tolist())
if not k_users:
    st.info("No Klinik users found in the selected **date range**.")
else:
    k_user = st.selectbox("Klinik staff member", k_users, key="klin_picker")
    kdf = klinik[klinik["person"]==k_user].copy()
    if kdf.empty:
        st.warning("No Klinik cases for this person with current filters. Try unticking **Limit to 08:00–18:30**.")
    else:
        kdf["hour"] = kdf["when"].dt.hour
        hours = list(range(8,19)) if clamp else sorted(kdf["hour"].unique().tolist())
        if not hours: hours = list(range(0,24))
        k_pivot = kdf.groupby("hour").size().reindex(hours, fill_value=0).reset_index()
        k_pivot.columns = ["Hour","Klinik Cases"]
        k_pivot["Hour"] = k_pivot["Hour"].apply(lambda h: f"{h:02d}:00")
        st.dataframe(k_pivot, use_container_width=True, height=260)
        with st.expander("Units closed in (top)", expanded=False):
            unit_tbl = kdf["unit"].value_counts().head(12).reset_index()
            unit_tbl.columns = ["Unit","Cases"]
            st.dataframe(unit_tbl, use_container_width=True)

st.markdown("---")

st.markdown("## Docman")
d_users = sorted(docman_all["person"].dropna().unique().tolist())
if not d_users:
    st.info("No Docman users found in the selected **date range**.")
else:
    d_user = st.selectbox("Docman staff member", d_users, key="doc_picker")
    ddf = docman[docman["person"]==d_user].copy()
    if ddf.empty:
        st.warning("No Docman tasks for this person with current filters. Try unticking **Limit to 08:00–18:30**.")
    else:
        ddf["hour"] = ddf["when"].dt.hour
        hours = list(range(8,19)) if clamp else sorted(ddf["hour"].unique().tolist())
        if not hours: hours = list(range(0,24))
        d_pivot = ddf.groupby("hour").size().reindex(hours, fill_value=0).reset_index()
        d_pivot.columns = ["Hour","Docman Completed"]
        d_pivot["Hour"] = d_pivot["Hour"].apply(lambda h: f"{h:02d}:00")
        st.dataframe(d_pivot, use_container_width=True, height=260)

st.markdown("---")

st.markdown("## Calls")
c_users = sorted(calls_all["person"].dropna().unique().tolist())
if not c_users:
    st.info("No Calls users found in the selected **date range**.")
else:
    c_user = st.selectbox("Calls staff member", c_users, key="call_picker", help="For callbacks, 'person' is the caller name; for inbound, it's the 'User Name'.")
    cdf = calls[calls["person"]==c_user].copy()
    if cdf.empty:
        if only_answered:
            st.warning("No calls for this person with current filters. Try unticking **Calls: only include answered/connected** or **Limit to 08:00–18:30**.")
        else:
            st.warning("No calls for this person with current filters. Try unticking **Limit to 08:00–18:30**.")
    else:
        cdf["hour"] = cdf["when"].dt.hour
        hours = list(range(8,19)) if clamp else sorted(cdf["hour"].unique().tolist())
        if not hours: hours = list(range(0,24))
        c_pivot = (cdf.groupby(["hour","kind"]).size()
                     .unstack(fill_value=0).reindex(index=hours, fill_value=0).reset_index())
        if "Callback" not in c_pivot.columns: c_pivot["Callback"]=0
        if "Group" not in c_pivot.columns: c_pivot["Group"]=0
        c_pivot["Total"] = c_pivot["Callback"] + c_pivot["Group"]
        c_pivot.rename(columns={"hour":"Hour"}, inplace=True)
        c_pivot["Hour"] = c_pivot["Hour"].apply(lambda h: f"{h:02d}:00")
        st.dataframe(c_pivot[["Hour","Group","Callback","Total"]], use_container_width=True, height=260)
        fig = go.Figure()
        fig.add_bar(x=c_pivot["Hour"], y=c_pivot["Group"], name="Group")
        fig.add_bar(x=c_pivot["Hour"], y=c_pivot["Callback"], name="Callback")
        fig.update_layout(barmode="stack", title="Calls by hour")
        st.plotly_chart(fig, use_container_width=True)
