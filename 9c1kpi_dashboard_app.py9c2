
import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

st.set_page_config(page_title="Modality Lewisham — A‑Team KPI Dashboard (v9c3)", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

# ---------- Branding ----------
st.markdown("""
<style>
:root { --ml-primary:#005eb8; --ml-accent:#00a3a3; --ml-muted:#f2f7ff; }
.ml-header{background:var(--ml-muted);border:1px solid #e6eefc;padding:14px 18px;border-radius:16px;display:flex;align-items:center;gap:14px;margin-bottom:10px;}
.ml-pill{background:var(--ml-primary);color:#fff;font-weight:700;padding:4px 10px;border-radius:999px;font-size:12px;letter-spacing:.3px;}
.ml-title{margin:0;font-weight:800;font-size:22px;color:#0b2e59;}
.ml-sub{margin:0;color:#345;font-size:13px;}
.stDataFrame{border-radius:12px;overflow:hidden;border:1px solid #eef3ff;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ml-header">
  <span class="ml-pill">Modality Lewisham</span>
  <div>
    <p class="ml-title">A‑Team KPI Dashboard</p>
    <p class="ml-sub">Week view → day → hour • Calls (Group + Callback), Docman, Klinik (with Unit drill-down)</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Auth ----------
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

# ---------- Helpers ----------
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

def norm_str(x):
    if pd.isna(x): return None
    s = str(x).strip()
    return s if s else None

def clean_token(s): return re.sub(r"[^a-z]", "", str(s).lower())

def email_candidates(email):
    if not isinstance(email, str): return set()
    local = email.split("@")[0].lower()
    for ch in [".","_","-","+"]: local = local.replace(ch," ")
    parts = [p for p in local.split() if p]
    keys=set()
    if parts:
        first, last = parts[0], parts[-1] if len(parts)>1 else ""
        if last: keys.add(clean_token(first[:1] + last))
        keys.add(clean_token("".join(parts)))
        keys.add(clean_token("".join(p[0] for p in parts)))
    else:
        keys.add(clean_token(local))
    return keys

def name_candidates(name):
    if not isinstance(name,str): return set()
    t = name.strip().lower()
    for ch in ["-","'","’",".","_","@"]: t = t.replace(ch," ")
    t = re.sub(r"\\s+"," ",t).strip()
    parts=[p for p in t.split(" ") if p and not p.isdigit()]
    keys=set()
    if parts:
        first,last = parts[0], parts[-1] if len(parts)>1 else ""
        if last: keys.add(clean_token(first[:1]+last))
        keys.add(clean_token("".join(parts)))
        keys.add(clean_token("".join(p[0] for p in parts)))
    else:
        keys.add(clean_token(t))
    return keys

def unify_people(*series_list):
    """Build a key→canonical mapping from *all* sources, not only emails.
       Canonical = first email seen; else the longest tokenised string.
    """
    key_to_canon = {}
    canon_pref = {}  # key -> (is_email, length, value)
    for ser in series_list:
        for raw in pd.Series(ser).dropna().astype(str):
            raw = raw.strip()
            keys = set()
            if "@" in raw: keys |= email_candidates(raw)
            keys |= name_candidates(raw)
            if not keys: continue
            is_email = "@" in raw
            score = (1 if is_email else 0, len(raw))
            for k in keys:
                best = canon_pref.get(k)
                if (best is None) or (score > best):
                    canon_pref[k] = score
                    key_to_canon[k] = raw.lower() if is_email else raw  # emails lower, names as-is
    return key_to_canon

def normalise_person(value, key_to_canon):
    s = norm_str(value)
    if not s: return None
    # prefer email lowercasing
    if "@" in s: return s.lower()
    for k in name_candidates(s):
        if k in key_to_canon:
            return key_to_canon[k]
    return s

def find_col(cols, targets=None, contains_any=None):
    targets = targets or []
    low=[c.lower() for c in cols]
    for t in targets:
        if t.lower() in low: return cols[low.index(t.lower())]
    if contains_any:
        for i,c in enumerate(low):
            if any(tok.lower() in c for tok in contains_any): return cols[i]
    return None

def find_datetime_col(df, prefer_tokens=None):
    prefer_tokens = prefer_tokens or []
    best=None
    for col in df.columns:
        ser=df[col]
        parsed=pd.to_datetime(ser, errors="coerce", dayfirst=True, infer_datetime_format=True)
        ok=parsed.notna().mean()
        if ok>=0.5:
            score=ok + (0.5 if any(tok in str(col).lower() for tok in prefer_tokens) else 0)
            if (best is None) or (score>best[0]): best=(score,col)
    return best[1] if best else None

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Filters")
    today=date.today()
    default_start=today - timedelta(days=today.weekday())
    default_end=default_start + timedelta(days=4)
    start=st.date_input("Start date", value=default_start)
    end=st.date_input("End date", value=default_end)
    if end<start: st.error("End date must be on or after start date"); st.stop()
    st.subheader("Upload files")
    case_file=st.file_uploader("Klinik Case Counts (CSV/XLSX)", type=["csv","xlsx","xls"])
    doc_file=st.file_uploader("Docman Tasks (CSV/XLSX)", type=["csv","xlsx","xls"])
    call_file=st.file_uploader("Telephone Calls Export (CSV/XLSX)", type=["csv","xlsx","xls"])
    only_answered=st.checkbox("Only include answered/connected calls", value=True)
    clamp = st.checkbox("Limit to 08:00–18:30", value=True)

case_df=load_table(case_file) if case_file else None
doc_df=load_table(doc_file) if doc_file else None
call_df=load_table(call_file) if call_file else None
if any(x is None for x in (case_df, doc_df, call_df)):
    st.info("Upload **all three** files (Klinik, Docman, Calls) to continue."); st.stop()

# ---------- Column mapping ----------
# Klinik
case_staff_col = find_col(case_df.columns, ["last_archived_by"], contains_any=["archived by","staff","user"])
case_date_col  = find_col(case_df.columns, ["last_archived_date"], contains_any=["archived date","date"])
case_time_col  = find_col(case_df.columns, ["last_archived_time"], contains_any=["archived time","time"])
case_unit_col  = find_col(case_df.columns, ["unit_closed_in","unit"], contains_any=["unit","closed in"])
missing_klinik=[n for n,v in {"Staff":case_staff_col,"Date":case_date_col,"Unit":case_unit_col}.items() if v is None]
if missing_klinik: st.error("Klinik file: couldn't auto-detect: " + ", ".join(missing_klinik)); st.stop()

# Docman
doc_user_col = find_col(doc_df.columns, ["User","Completed User"], contains_any=["user","completed"])
doc_dt_col   = find_col(doc_df.columns, ["Date and Time of Event"], contains_any=["date and time of event","completed","datetime","date","time"])
if doc_dt_col is None: doc_dt_col = find_datetime_col(doc_df, ["date","time","event","completed"])
missing_doc=[n for n,v in {"User":doc_user_col,"DateTime":doc_dt_col}.items() if v is None]
if missing_doc: st.error("Docman file: couldn't auto-detect: " + ", ".join(missing_doc)); st.stop()

# Calls
call_user_answered_col = find_col(call_df.columns, ["User Name","Answered","Answered By","Agent"], contains_any=["user name","answered","agent","owner","user"])
call_caller_col        = find_col(call_df.columns, ["Caller Name","Caller"], contains_any=["caller","callback"])
call_type_col          = find_col(call_df.columns, ["Call Type","Type"], contains_any=["type","call type","group","callback","cb"])
call_outcome_col       = find_col(call_df.columns, ["Outcome"], contains_any=["outcome","result","status","disposition"])
call_start_col         = find_col(call_df.columns, ["Start Time","Start","StartDateTime","Start Datetime","Call Start Time","Call Started"], contains_any=["start time","start","begin"])
if call_start_col is None: call_start_col = find_datetime_col(call_df, ["start","begin","created","call","time","date"])
missing_calls=[n for n,v in {"Answered/User":call_user_answered_col,"Caller":call_caller_col,"Call Type":call_type_col,"Outcome":call_outcome_col,"Start":call_start_col}.items() if v is None]
if missing_calls: st.error("Calls file: couldn't auto-detect: " + ", ".join(missing_calls)); st.stop()

# ---------- Build identity index from all sources ----------
key_to_canon = unify_people(case_df[case_staff_col], doc_df[doc_user_col], call_df[call_user_answered_col], call_df[call_caller_col])

# ---------- Build event frames ----------
# Klinik
case_dt = parse_dt(case_df[case_date_col], case_df[case_time_col] if case_time_col in case_df.columns else None) if case_time_col else parse_dt(case_df[case_date_col])
case_person = case_df[case_staff_col].apply(lambda v: normalise_person(v, key_to_canon))
case_units = case_df[case_unit_col].astype(str).replace(["nan","None",""], np.nan).fillna("Unknown")
klinik_ev = pd.DataFrame({"when": case_dt, "person": case_person, "kind": "Klinik Case", "unit": case_units})

# Docman
doc_dt = parse_dt(doc_df[doc_dt_col], None)
doc_person = doc_df[doc_user_col].apply(lambda v: normalise_person(v, key_to_canon))
doc_ev = pd.DataFrame({"when": doc_dt, "person": doc_person, "kind": "Docman Completed", "unit": np.nan})

# Calls
dt = parse_dt(call_df[call_start_col], None)
type_lower = call_df[call_type_col].astype(str).str.lower()
out_lower  = call_df[call_outcome_col].astype(str).str.lower()
is_callback = type_lower.str.contains("callback|call back|cb", na=False)
who_calls_raw = np.where(is_callback, call_df[call_caller_col].astype(str), call_df[call_user_answered_col].astype(str))
who_calls = pd.Series(who_calls_raw).apply(lambda v: normalise_person(v, key_to_canon))
calls_kind = np.where(is_callback, "Call — Callback", "Call — Group")
calls_ev = pd.DataFrame({"when": dt, "person": who_calls, "kind": calls_kind, "unit": np.nan})
if only_answered:
    answered_mask = out_lower.str.contains("answer|answered|connect|connected|complete|completed|handled|finished|resolved|success|ok", na=False)
    calls_ev = calls_ev[answered_mask]

# Combine
df = pd.concat([klinik_ev, doc_ev, calls_ev], ignore_index=True).dropna(subset=["when","person"])

# Filter by date range
start_dt = datetime.combine(start, datetime.min.time())
end_dt = datetime.combine(end, datetime.max.time())
df = df[(df["when"] >= pd.Timestamp(start_dt)) & (df["when"] <= pd.Timestamp(end_dt))].copy()

# Optional time clamp
if clamp:
    df = df[((df["when"].dt.hour > 8) | ((df["when"].dt.hour == 8) & (df["when"].dt.minute >= 0))) &
            ((df["when"].dt.hour < 18) | ((df["when"].dt.hour == 18) & (df["when"].dt.minute <= 30)))]

if df.empty:
    st.warning("No events for the selected filters."); st.stop()

# Staff picker
with st.sidebar:
    users = sorted(df["person"].dropna().unique().tolist())
    selected_user = st.selectbox("Staff Member", users)

udf = df[df["person"] == selected_user].copy()
if udf.empty:
    st.warning("No events for this person in the selected range."); st.stop()

# Derive day/hour
udf["date_only"] = udf["when"].dt.date
udf["hour"] = udf["when"].dt.hour

# KPI
def kpi_row(data):
    total = int(len(data))
    calls_total = int((data["kind"].astype(str).str.startswith("Call —")).sum())
    doc_total = int((data["kind"]=="Docman Completed").sum())
    klinik_total = int((data["kind"]=="Klinik Case").sum())
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total tasks", f"{total}")
    c2.metric("Calls (Group + Callback)", f"{calls_total}")
    c3.metric("Docman Completed", f"{doc_total}")
    c4.metric("Klinik Cases", f"{klinik_total}")
kpi_row(udf)

# Week summary
week = []
for d,g in udf.groupby("date_only"):
    grp = (g["kind"]=="Call — Group").sum()
    cb  = (g["kind"]=="Call — Callback").sum()
    doc = (g["kind"]=="Docman Completed").sum()
    kli = (g["kind"]=="Klinik Case").sum()
    label = pd.Timestamp(d).strftime("%a %d %b")
    week.append({"Date":label,"Calls: Group":int(grp),"Calls: Callback":int(cb),"Calls Total":int(grp+cb),
                 "Docman":int(doc),"Klinik":int(kli),"Grand Total":int(grp+cb+doc+kli),"key":d})
week_df = pd.DataFrame(week).sort_values("key")
if not week_df.empty:
    total_row = {"Date":"Total","Calls: Group":int(week_df["Calls: Group"].sum()),
                 "Calls: Callback":int(week_df["Calls: Callback"].sum()),
                 "Calls Total":int(week_df["Calls Total"].sum()),
                 "Docman":int(week_df["Docman"].sum()),
                 "Klinik":int(week_df["Klinik"].sum()),
                 "Grand Total":int(week_df["Grand Total"].sum())}
    week_df = pd.concat([week_df, pd.DataFrame([total_row])], ignore_index=True)
st.markdown("### Week summary — by day")
st.dataframe(week_df.drop(columns=["key"], errors="ignore"), use_container_width=True, height=260)

# Day selector
valid_days = sorted(udf["date_only"].unique().tolist())
with st.sidebar:
    pick_day = st.selectbox("Drill down day", [pd.Timestamp(d).strftime("%A %d %B") for d in valid_days], index=0)
pick_day_dt = datetime.strptime(pick_day, "%A %d %B").date()
dr = udf[udf["date_only"]==pick_day_dt].copy()

# Robust hourly grid via pivot
hours = list(range(8,19)) if clamp else sorted(dr["hour"].dropna().unique())
if not hours: hours = list(range(0,24))

cats = ["Call — Group","Call — Callback","Docman Completed","Klinik Case"]
dr["kind"] = dr["kind"].astype(str)
pivot = (dr.assign(kind_clean=dr["kind"].where(dr["kind"].isin(cats),"Other"))
           .groupby(["hour","kind_clean"]).size().unstack(fill_value=0)
           .reindex(index=hours, fill_value=0))
for need in ["Call — Group","Call — Callback","Docman Completed","Klinik Case"]:
    if need not in pivot.columns: pivot[need]=0
hour_grid = pd.DataFrame({
    "Hour":[f"{h:02d}:00" for h in pivot.index],
    "Calls (Group+Callback)": (pivot["Call — Group"] + pivot["Call — Callback"]).astype(int),
    "Docman Completed": pivot["Docman Completed"].astype(int),
    "Klinik Case": pivot["Klinik Case"].astype(int)
})
hour_grid["Total"] = hour_grid["Calls (Group+Callback)"] + hour_grid["Docman Completed"] + hour_grid["Klinik Case"]

st.markdown(f"### {selected_user} — {pick_day} {'(08:00–18:30)' if clamp else '(all hours)'}")
st.dataframe(hour_grid, use_container_width=True, height=360)

# Stacked hour chart
fig = go.Figure()
fig.add_bar(x=hour_grid["Hour"], y=hour_grid["Calls (Group+Callback)"], name="Calls (Group+Callback)")
fig.add_bar(x=hour_grid["Hour"], y=hour_grid["Docman Completed"], name="Docman Completed")
fig.add_bar(x=hour_grid["Hour"], y=hour_grid["Klinik Case"], name="Klinik Case")
fig.update_layout(barmode="stack", title=f"{pick_day}", xaxis_title="Hour", yaxis_title="Count", legend_title="Category")
st.plotly_chart(fig, use_container_width=True)

# Unit drill-down
with st.expander("Klinik — drill down by Unit (for this staff + day)", expanded=False):
    hour_options = ["All hours"] + [f"{h:02d}:00" for h in pivot.index]
    hour_choice = st.selectbox("Hour filter (optional)", hour_options, index=0)
    if hour_choice == "All hours":
        klin_sel = dr[dr["kind"]=="Klinik Case"]
    else:
        hh = int(hour_choice.split(":")[0])
        klin_sel = dr[(dr["kind"]=="Klinik Case") & (dr["hour"]==hh)]
    if klin_sel.empty:
        st.info("No Klinik cases for the selected staff/day/hour.")
    else:
        unit_tbl = (klin_sel["unit"].dropna().astype(str).value_counts()
                    .reset_index().rename(columns={"index":"Unit","count":"Cases"}))
        unit_tbl.columns = ["Unit","Cases"]
        st.dataframe(unit_tbl, use_container_width=True, height=260)
        st.download_button("Download unit breakdown (CSV)", data=unit_tbl.to_csv(index=False).encode("utf-8"),
                           file_name="unit_breakdown.csv", mime="text/csv")

# Detail panes (use the same pivot pieces)
c1,c2,c3 = st.columns(3)
with c1:
    st.caption("Calls — Group vs Callback (by hour)")
    calls_tbl = pivot[["Call — Group","Call — Callback"]].reset_index().rename(columns={"hour":"Hour"})
    calls_tbl["Hour"] = calls_tbl["Hour"].apply(lambda h: f"{h:02d}:00")
    calls_tbl["Total"] = calls_tbl["Call — Group"] + calls_tbl["Call — Callback"]
    st.dataframe(calls_tbl, use_container_width=True, height=280)

with c2:
    st.caption("Klinik — Units closed in (top for day)")
    klin_day = dr[dr["kind"]=="Klinik Case"]
    if klin_day.empty:
        st.info("No Klinik cases for this staff/day.")
    else:
        top_units = (klin_day["unit"].dropna().astype(str).value_counts().head(8).reset_index())
        top_units.columns = ["Unit","Cases"]
        st.dataframe(top_units, use_container_width=True, height=280)

with c3:
    st.caption("Docman — per hour")
    doc_tbl = pivot["Docman Completed"].astype(int).reset_index()
    doc_tbl.columns = ["Hour","Docman Completed"]
    doc_tbl["Hour"] = doc_tbl["Hour"].apply(lambda h: f"{h:02d}:00")
    st.dataframe(doc_tbl, use_container_width=True, height=280)

# Diagnostics
with st.expander("Diagnostics", expanded=False):
    st.write({
        "Klinik": {"staff": case_staff_col, "date": case_date_col, "time": case_time_col, "unit": case_unit_col},
        "Docman": {"user": doc_user_col, "datetime": doc_dt_col},
        "Calls":  {"answered/user": call_user_answered_col, "caller": call_caller_col, "type": call_type_col, "outcome": call_outcome_col, "start": call_start_col},
    })
