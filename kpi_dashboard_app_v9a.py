
import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

st.set_page_config(page_title="Modality Lewisham — A‑Team KPI Dashboard (v9a)", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root { --ml-primary:#005eb8; --ml-accent:#00a3a3; --ml-muted:#f2f7ff; }
    .ml-header{background:var(--ml-muted);border:1px solid #e6eefc;padding:14px 18px;border-radius:16px;display:flex;align-items:center;gap:14px;margin-bottom:10px;}
    .ml-pill{background:var(--ml-primary);color:#fff;font-weight:700;padding:4px 10px;border-radius:999px;font-size:12px;letter-spacing:.3px;}
    .ml-title{margin:0;font-weight:800;font-size:22px;color:#0b2e59;}
    .ml-sub{margin:0;color:#345;font-size:13px;}
    .stDataFrame{border-radius:12px;overflow:hidden;border:1px solid #eef3ff;}
    </style>
    """, unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="ml-header">
      <span class="ml-pill">Modality Lewisham</span>
      <div>
        <p class="ml-title">A‑Team KPI Dashboard</p>
        <p class="ml-sub">Week view → drill down by day/hour • Calls (Group + Callback), Docman Completed, Klinik Cases</p>
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
            file.seek(0)
            return pd.read_excel(file)
    for kwargs in [{"sep": ",", "encoding": "utf-8"},
                   {"sep": ";", "encoding": "utf-8"},
                   {"sep": ",", "encoding": "latin-1"},
                   {"sep": ";", "encoding": "latin-1"}]:
        try:
            file.seek(0); return pd.read_csv(file, **kwargs)
        except Exception:
            continue
    file.seek(0); return pd.read_csv(file)

def parse_dt(date_series, time_series=None):
    if time_series is None:
        return pd.to_datetime(date_series, errors="coerce", dayfirst=True, infer_datetime_format=True)
    s = date_series.astype(str).strip() + " " + time_series.astype(str).strip()
    return pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)

def norm_email_or_name(x):
    if pd.isna(x): return None
    s = str(x).strip()
    if "@" in s: return s.lower()
    return " ".join(w.capitalize() for w in s.split() if w)

def find_col(cols, targets=None, contains_any=None):
    targets = targets or []
    low = [c.lower() for c in cols]
    for t in targets:
        if t.lower() in low: return cols[low.index(t.lower())]
    if contains_any:
        for i, c in enumerate(low):
            if any(tok.lower() in c for tok in contains_any): return cols[i]
    return None

def clean_token(s): return re.sub(r"[^a-z]", "", s.lower())

def email_candidates(email):
    if not isinstance(email, str): return set()
    local = email.split("@")[0].lower()
    # safe split on common separators
    for ch in [".", "_", "-", "+"]:
        local = local.replace(ch, " ")
    parts = [p for p in local.split() if p]
    keys = set()
    if parts:
        first, last = parts[0], parts[-1] if len(parts)>1 else ""
        if last: keys.add(clean_token(first[:1] + last))
        keys.add(clean_token("".join(parts)))
        keys.add(clean_token("".join(parts)))  # already condensed
        initials = "".join(p[:1] for p in parts); keys.add(clean_token(initials))
    else:
        keys.add(clean_token(local))
    return keys

def name_candidates(name):
    """Regex-free tokeniser to avoid PatternError on some Python builds."""
    if not isinstance(name, str): return set()
    t = name.strip().lower()
    for ch in ["-", "'", "’", ".", "_"]:
        t = t.replace(ch, " ")
    t = re.sub(r"\s+", " ", t).strip()
    parts = [p for p in t.split(" ") if p]
    keys = set()
    if parts:
        first, last = parts[0], parts[-1] if len(parts)>1 else ""
        if last: keys.add(clean_token(first[:1] + last))
        keys.add(clean_token("".join(parts)))
        initials = "".join(p[:1] for p in parts); keys.add(clean_token(initials))
    else:
        keys.add(clean_token(t))
    return keys

def build_email_matcher(klinik_emails):
    key_to_email = {}; collisions = set()
    for em in sorted(set([e for e in klinik_emails if isinstance(e, str) and "@" in e])):
        for key in email_candidates(em):
            if key in key_to_email and key_to_email[key] != em: collisions.add(key)
            else: key_to_email[key] = em
    for k in collisions: key_to_email.pop(k, None)
    return key_to_email, collisions

def find_datetime_col(df, prefer_tokens=None):
    prefer_tokens = prefer_tokens or []
    candidates = []
    for col in df.columns:
        ser = df[col]
        if ser.dtype == "object" or "datetime" in str(ser.dtype).lower() or "date" in str(ser.dtype).lower() or "int" in str(ser.dtype).lower() or "float" in str(ser.dtype).lower():
            parsed = pd.to_datetime(ser, errors="coerce", dayfirst=True, infer_datetime_format=True)
            ok = parsed.notna().mean()
            if ok >= 0.5:
                name_lower = str(col).lower()
                score = ok + (0.5 if any(tok in name_lower for tok in prefer_tokens) else 0.0)
                candidates.append((score, col))
    if not candidates: return None
    candidates.sort(reverse=True); return candidates[0][1]

with st.sidebar:
    st.header("Filters")
    today = date.today()
    default_start = today - timedelta(days=today.weekday())
    default_end = default_start + timedelta(days=4)
    start = st.date_input("Start date", value=default_start, key="start_date")
    end = st.date_input("End date", value=default_end, key="end_date")
    if end < start: st.error("End date must be on or after start date"); st.stop()
    st.subheader("Upload files")
    case_file = st.file_uploader("Klinik Case Counts (CSV/XLSX)", type=["csv","xlsx","xls"], key="case")
    doc_file  = st.file_uploader("Docman Tasks (CSV/XLSX)", type=["csv","xlsx","xls"], key="docman")
    call_file = st.file_uploader("Telephone Calls Export (CSV/XLSX)", type=["csv","xlsx","xls"], key="calls")

case_df = load_table(case_file) if case_file else None
doc_df  = load_table(doc_file) if doc_file else None
call_df = load_table(call_file) if call_file else None
if any(x is None for x in (case_df, doc_df, call_df)):
    st.info("Upload **all three** files (Klinik, Docman, Calls) to continue."); st.stop()

# Auto-map columns
case_cols = case_df.columns
case_staff_col = find_col(case_cols, ["last_archived_by"], contains_any=["archived by","archived_by","staff","user"])
case_date_col  = find_col(case_cols, ["last_archived_date"], contains_any=["archived date","archived_date","date"])
case_time_col  = find_col(case_cols, ["last_archived_time"], contains_any=["archived time","archived_time","time"])
case_unit_col  = find_col(case_cols, [], contains_any=["unit","unit_closed","unit closed","closed in","closed_in"])
missing_klinik = [n for n,v in {"Staff":case_staff_col,"Date":case_date_col,"Time":case_time_col,"Unit":case_unit_col}.items() if v is None]
if missing_klinik: st.error("Klinik file: couldn't auto-detect: " + ", ".join(missing_klinik)); st.stop()

doc_cols = doc_df.columns
doc_user_col = find_col(doc_cols, ["User","Completed User","Completed_User","Completed By","Completed_by"], contains_any=["user","completed"])
doc_dt_col   = find_col(doc_cols, ["Date and Time of Event"], contains_any=["date and time of event","completed","datetime","date","time of event","time"])
if doc_dt_col is None: doc_dt_col = find_datetime_col(doc_df, prefer_tokens=["date","time","event","completed"])
missing_doc = [n for n,v in {"User":doc_user_col,"DateTime":doc_dt_col}.items() if v is None]
if missing_doc: st.error("Docman file: couldn't auto-detect: " + ", ".join(missing_doc)); st.stop()

call_cols = call_df.columns
call_user_answered_col = find_col(call_cols, ["User Name","Answered","Answered By","Agent"], contains_any=["user name","answered","agent","owner","user"])
call_caller_col        = find_col(call_cols, ["Caller Name","Caller"], contains_any=["caller","callback"])
call_type_col          = find_col(call_cols, ["Call Type","Type"], contains_any=["type","call type","group","callback","cb"])
call_outcome_col       = find_col(call_cols, ["Outcome"], contains_any=["outcome","result","status","disposition"])
call_start_col         = find_col(call_cols, ["Start Time","Start","StartDateTime","Start Datetime","Call Start Time","Call Started"], contains_any=["start time","started","start","datetime","date","time"])
if call_start_col is None: call_start_col = find_datetime_col(call_df, prefer_tokens=["start","begin","created","call","time","date"])
missing_calls = [n for n,v in {"Answered/User":call_user_answered_col,"Caller":call_caller_col,"Call Type":call_type_col,"Outcome":call_outcome_col,"Start":call_start_col}.items() if v is None]
if missing_calls: st.error("Calls file: couldn't auto-detect: " + ", ".join(missing_calls)); st.stop()

# Events & identity
case_dt = parse_dt(case_df[case_date_col], case_df[case_time_col])
case_staff_email = case_df[case_staff_col].map(norm_email_or_name)
case_units = case_df[case_unit_col].astype(str).replace(["nan","None"], np.nan).fillna("Unknown")
klinik_ev = pd.DataFrame({"when": case_dt, "person": case_staff_email, "kind": "Klinik Case", "unit": case_units})

key_to_email, collisions = build_email_matcher(case_staff_email.dropna())

def best_email_for_name_or_email(value):
    s = str(value) if not pd.isna(value) else ""
    if "@" in s: return s.lower()
    for k in name_candidates(s):
        if k in key_to_email: return key_to_email[k]
    return norm_email_or_name(value)

doc_dt = parse_dt(doc_df[doc_dt_col], None)
doc_user_norm = doc_df[doc_user_col].apply(best_email_for_name_or_email)
doc_ev = pd.DataFrame({"when": doc_dt, "person": doc_user_norm, "kind": "Docman Completed", "unit": np.nan})

dt = parse_dt(call_df[call_start_col], None)
type_lower = call_df[call_type_col].astype(str).str.lower()
out_lower  = call_df[call_outcome_col].astype(str).str.lower()
is_callback = type_lower.str.contains("callback|call back|cb")
who_calls = np.where(is_callback, call_df[call_caller_col].astype(str), call_df[call_user_answered_col].astype(str))
who_calls = pd.Series(who_calls).apply(best_email_for_name_or_email)
calls_kind = np.where(is_callback, "Call — Callback", "Call — Group")
answered_mask = out_lower.str.contains("answer|connect|complete|handled|finished|resolved|success|ok")
calls_ev = pd.DataFrame({"when": dt, "person": who_calls, "kind": calls_kind, "unit": np.nan})
try: calls_ev = calls_ev[answered_mask.values]
except Exception: pass

df = pd.concat([klinik_ev, doc_ev, calls_ev], ignore_index=True).dropna(subset=["when","person"])

start_dt = datetime.combine(start, datetime.min.time())
end_dt   = datetime.combine(end, datetime.max.time())
df = df[(df["when"] >= pd.Timestamp(start_dt)) & (df["when"] <= pd.Timestamp(end_dt))].copy()

df = df[((df["when"].dt.hour > 8) | ((df["when"].dt.hour == 8) & (df["when"].dt.minute >= 0))) &
        ((df["when"].dt.hour < 18) | ((df["when"].dt.hour == 18) & (df["when"].dt.minute <= 30)))]

if df.empty: st.warning("No events for the selected date range/time window."); st.stop()

with st.sidebar:
    users = sorted(df["person"].dropna().unique().tolist())
    selected_user = st.selectbox("Staff Member", users)

if not selected_user: st.info("Choose a staff member to continue."); st.stop()

udf = df[df["person"] == selected_user].copy()
if udf.empty: st.warning("No events for this person in the selected range."); st.stop()

udf["day_label"] = udf["when"].dt.strftime("%a %d %b")
udf["date_only"] = udf["when"].dt.date

def kpi_row():
    total = int(len(udf))
    calls_total = int((udf["kind"].astype(str).str.startswith("Call —")).sum())
    doc_total   = int((udf["kind"]=="Docman Completed").sum())
    klinik_total= int((udf["kind"]=="Klinik Case").sum())
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total tasks", f"{total}")
    c2.metric("Calls (Group + Callback)", f"{calls_total}")
    c3.metric("Docman Completed", f"{doc_total}")
    c4.metric("Klinik Cases", f"{klinik_total}")
kpi_row()

week = []
for d, g in udf.groupby("date_only"):
    grp_calls = (g["kind"]=="Call — Group").sum()
    cb_calls  = (g["kind"]=="Call — Callback").sum()
    doc_cnt   = (g["kind"]=="Docman Completed").sum()
    klinik_cnt= (g["kind"]=="Klinik Case").sum()
    label = pd.Timestamp(d).strftime("%a %d %b")
    week.append({"Date": label, "Calls: Group": int(grp_calls), "Calls: Callback": int(cb_calls),
                 "Calls Total": int(grp_calls+cb_calls), "Docman": int(doc_cnt), "Klinik": int(klinik_cnt),
                 "Grand Total": int(grp_calls+cb_calls+doc_cnt+klinik_cnt), "key": d})
week_df = pd.DataFrame(week).sort_values("key")
if not week_df.empty:
    totals = {"Date":"Total","Calls: Group": int(week_df["Calls: Group"].sum()),
              "Calls: Callback": int(week_df["Calls: Callback"].sum()),
              "Calls Total": int(week_df["Calls Total"].sum()),
              "Docman": int(week_df["Docman"].sum()),
              "Klinik": int(week_df["Klinik"].sum()),
              "Grand Total": int(week_df["Grand Total"].sum()),
              "key": pd.NaT}
    week_df = pd.concat([week_df, pd.DataFrame([totals])], ignore_index=True)
st.markdown("### Week summary — by day")
st.dataframe(week_df.drop(columns=["key"]), use_container_width=True, height=260)

valid_days = sorted(udf["date_only"].unique().tolist())
with st.sidebar:
    pick_day = st.selectbox("Drill down day", [pd.Timestamp(d).strftime("%A %d %B") for d in valid_days], index=0)
pick_day_dt = datetime.strptime(pick_day, "%A %d %B").date()

dr = udf[udf["date_only"]==pick_day_dt].copy()
dr["hour"] = dr["when"].dt.hour

hours = list(range(8,19))
rows = []
for h in hours:
    row = {"Hour": f"{h:02d}:00"}
    row["Calls (Group+Callback)"] = int((((dr["kind"]=="Call — Group") | (dr["kind"]=="Call — Callback")) & (dr["hour"]==h)).sum())
    row["Docman Completed"] = int(((dr["kind"]=="Docman Completed") & (dr["hour"]==h)).sum())
    row["Klinik Case"] = int(((dr["kind"]=="Klinik Case") & (dr["hour"]==h)).sum())
    row["Total"] = row["Calls (Group+Callback)"] + row["Docman Completed"] + row["Klinik Case"]
    rows.append(row)
hour_grid = pd.DataFrame(rows)

st.markdown(f"### {selected_user} — {pick_day} (08:00–18:30)")
st.dataframe(hour_grid, use_container_width=True, height=360)

fig = go.Figure()
fig.add_bar(x=hour_grid["Hour"], y=hour_grid["Calls (Group+Callback)"], name="Calls (Group+Callback)")
fig.add_bar(x=hour_grid["Hour"], y=hour_grid["Docman Completed"], name="Docman Completed")
fig.add_bar(x=hour_grid["Hour"], y=hour_grid["Klinik Case"], name="Klinik Case")
fig.update_layout(barmode="stack", title=f"{pick_day}", xaxis_title="Hour", yaxis_title="Count", legend_title="Category")
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Day details")
c1,c2,c3 = st.columns(3)
calls_day = dr[dr["kind"].str.startswith("Call —")]
calls_split = []
for h in hours:
    g = int((((calls_day["kind"]=="Call — Group") & (calls_day["hour"]==h)).sum()))
    cb = int((((calls_day["kind"]=="Call — Callback") & (calls_day["hour"]==h)).sum()))
    calls_split.append({"Hour": f"{h:02d}:00", "Group": g, "Callback": cb, "Total": g+cb})
with c1:
    st.caption("Calls — Group vs Callback (by hour)")
    st.dataframe(pd.DataFrame(calls_split), use_container_width=True, height=280)

klin_day = dr[dr["kind"]=="Klinik Case"]
top_units = klin_day["unit"].dropna().astype(str).value_counts().head(8).reset_index()
top_units.columns = ["Unit", "Cases"]
with c2:
    st.caption("Klinik — Units closed in (top)")
    st.dataframe(top_units, use_container_width=True, height=280)

doc_day = dr[dr["kind"]=="Docman Completed"]
doc_hour = doc_day.groupby("hour").size().reindex(hours, fill_value=0).reset_index()
doc_hour.columns = ["Hour", "Docman Completed"]
doc_hour["Hour"] = doc_hour["Hour"].apply(lambda h: f"{h:02d}:00")
with c3:
    st.caption("Docman — per hour")
    st.dataframe(doc_hour, use_container_width=True, height=280)

with st.expander("Diagnostics", expanded=False):
    st.write({
        "Klinik": {"staff": case_staff_col, "date": case_date_col, "time": case_time_col, "unit": case_unit_col},
        "Docman": {"user": doc_user_col, "datetime": doc_dt_col},
        "Calls":  {"answered/user": call_user_answered_col, "caller": call_caller_col, "type": call_type_col, "outcome": call_outcome_col, "start": call_start_col},
    })
