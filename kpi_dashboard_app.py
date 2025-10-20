
import io
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

st.set_page_config(page_title="A-Team KPI Dashboard", layout="wide")

st.title("A-Team KPI Dashboard")
st.caption("Upload the three reports, pick your dates, and filter by staff. The dashboard will do the rest.")

with st.expander("📄 What you need to upload", expanded=True):
    st.markdown("""
    Upload **three** CSV files:
    1. **Klinik Case Counts** — each row is a case; a *closed case* counts as a task.  
       Typical columns: `last_archived_by`, `last_archived_date`, `last_archived_time`.
    2. **Docman10 Tasks** — each row is a completed task.  
       Typical columns: `Completed User`, `Date and Time of Event`.
    3. **Telephone Calls Export** — answered inbound calls and callbacks.  
       Typical columns: `User Name`, `Caller Name`, `Outcome`, `Direction`, `Start Time`.
    """)

# --- File uploaders
col_u1, col_u2, col_u3 = st.columns(3)

with col_u1:
    case_file = st.file_uploader("Klinik Case Counts (CSV)", type=["csv"], key="case")
with col_u2:
    docman_file = st.file_uploader("Docman10 Tasks (CSV)", type=["csv"], key="docman")
with col_u3:
    calls_file = st.file_uploader("Telephone Calls Export (CSV)", type=["csv"], key="calls")

# --- Date range selection
today = date.today()
default_start = today - timedelta(days=today.weekday())  # Monday of this week
default_end = default_start + timedelta(days=4)          # Friday

st.markdown("### 📆 Date range")
start_date, end_date = st.columns(2)
with start_date:
    start = st.date_input("Start date", value=default_start)
with end_date:
    end = st.date_input("End date", value=default_end)

if end < start:
    st.error("End date must be on or after the start date.")
    st.stop()

# Helper: robust datetime parsing
def parse_dt(date_series, time_series=None):
    if time_series is None:
        return pd.to_datetime(date_series, errors="coerce", dayfirst=True, utc=False)
    # combine
    combined = date_series.astype(str).str.strip() + " " + time_series.astype(str).str.strip()
    # try multiple common formats
    dt = pd.to_datetime(combined, errors="coerce", dayfirst=True, utc=False)
    return dt

def standardise_staff(name):
    if pd.isna(name):
        return None
    s = str(name).strip()
    # normalise email-like usernames to lowercase
    if "@" in s:
        return s.lower()
    # title case for names
    return " ".join([w.capitalize() for w in s.split() if w])

# --- Load & normalise datasets
events = []  # unified list of (when, who, source)

# Cases
if case_file is not None:
    try:
        case_df = pd.read_csv(case_file)
    except Exception:
        case_file.seek(0)
        case_df = pd.read_csv(case_file, sep=";")
    # Let user map columns
    st.markdown("#### 🗂️ Map fields — Klinik Cases")
    c1, c2, c3 = st.columns(3)
    with c1:
        col_archived_by = st.selectbox("Staff column", options=case_df.columns.tolist(), index=(list(case_df.columns.str.lower()).index("last_archived_by") if "last_archived_by" in case_df.columns.str.lower().tolist() else 0))
    with c2:
        col_archived_date = st.selectbox("Date column", options=case_df.columns.tolist(), index=(list(case_df.columns.str.lower()).index("last_archived_date") if "last_archived_date" in case_df.columns.str.lower().tolist() else 0))
    with c3:
        col_archived_time = st.selectbox("Time column", options=case_df.columns.tolist(), index=(list(case_df.columns.str.lower()).index("last_archived_time") if "last_archived_time" in case_df.columns.str.lower().tolist() else 0))
    # build events
    case_dt = parse_dt(case_df[col_archived_date], case_df[col_archived_time])
    case_staff = case_df[col_archived_by].map(standardise_staff)
    case_events = pd.DataFrame({"when": case_dt, "who": case_staff})
    case_events["source"] = "Klinik Cases"
    events.append(case_events)

# Docman
if docman_file is not None:
    try:
        doc_df = pd.read_csv(docman_file)
    except Exception:
        docman_file.seek(0)
        doc_df = pd.read_csv(docman_file, sep=";")
    st.markdown("#### 🗂️ Map fields — Docman10 Tasks")
    d1, d2 = st.columns(2)
    with d1:
        col_completed_user = st.selectbox("Completed user column", options=doc_df.columns.tolist(),
                                          index=(list(doc_df.columns.str.lower()).index("completed user") if "completed user" in doc_df.columns.str.lower().tolist() else 0))
    with d2:
        col_completed_dt = st.selectbox("Completed date/time column", options=doc_df.columns.tolist(),
                                        index=(list(doc_df.columns.str.lower()).index("date and time of event") if "date and time of event" in doc_df.columns.str.lower().tolist() else 0))
    doc_dt = parse_dt(doc_df[col_completed_dt], None)
    doc_staff = doc_df[col_completed_user].map(standardise_staff)
    doc_events = pd.DataFrame({"when": doc_dt, "who": doc_staff})
    doc_events["source"] = "Docman Tasks"
    events.append(doc_events)

# Calls
if calls_file is not None:
    try:
        calls_df = pd.read_csv(calls_file)
    except Exception:
        calls_file.seek(0)
        calls_df = pd.read_csv(calls_file, sep=";")
    st.markdown("#### 🗂️ Map fields — Telephone Calls")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        col_username = st.selectbox("Inbound staff column (User Name)", options=calls_df.columns.tolist(),
                                    index=(list(calls_df.columns.str.lower()).index("user name") if "user name" in calls_df.columns.str.lower().tolist() else 0))
    with c2:
        col_callername = st.selectbox("Callback staff column (Caller Name)", options=calls_df.columns.tolist(),
                                      index=(list(calls_df.columns.str.lower()).index("caller name") if "caller name" in calls_df.columns.str.lower().tolist() else 0))
    with c3:
        col_outcome = st.selectbox("Outcome column", options=calls_df.columns.tolist(),
                                   index=(list(calls_df.columns.str.lower()).index("outcome") if "outcome" in calls_df.columns.str.lower().tolist() else 0))
    with c4:
        col_direction = st.selectbox("Direction column", options=calls_df.columns.tolist(),
                                     index=(list(calls_df.columns.str.lower()).index("direction") if "direction" in calls_df.columns.str.lower().tolist() else 0))
    with c5:
        col_start = st.selectbox("Start time column", options=calls_df.columns.tolist(),
                                 index=(list(calls_df.columns.str.lower()).index("start time") if "start time" in calls_df.columns.str.lower().tolist() else 0))

    # Determine 'who': prefer User Name for inbound, else Caller Name for callbacks
    calls = calls_df.copy()
    call_dt = parse_dt(calls[col_start], None)
    direction_lower = calls[col_direction].astype(str).str.lower()
    outcome_lower = calls[col_outcome].astype(str).str.lower()

    answered_mask = outcome_lower.str.contains("answer|connect|complete|handled", regex=True)
    inbound_mask = direction_lower.str.contains("inbound")

    staff_series = np.where(inbound_mask,
                            calls[col_username].astype(str),
                            calls[col_callername].astype(str))
    staff_series = pd.Series(staff_series).map(standardise_staff)

    call_events = pd.DataFrame({"when": call_dt, "who": staff_series})
    call_events["source"] = "Calls (Answered)"

    try:
        call_events = call_events[answered_mask.values]
    except Exception:
        pass

    events.append(call_events)

# --- Combine & filter by date
if not events:
    st.info("Upload at least one file to see results.")
    st.stop()

all_events = pd.concat(events, ignore_index=True)
all_events = all_events.dropna(subset=["when", "who"])
# filter to inclusive date range for local time (no tz assumed)
start_dt = datetime.combine(start, datetime.min.time())
end_dt = datetime.combine(end, datetime.max.time())

mask = (all_events["when"] >= pd.Timestamp(start_dt)) & (all_events["when"] <= pd.Timestamp(end_dt))
all_events = all_events.loc[mask].copy()

if all_events.empty:
    st.warning("No events found in the selected date range.")
    st.stop()

# --- Staff filter
staff_list = sorted(all_events["who"].dropna().unique().tolist())
selected_staff = st.multiselect("Filter by staff (leave empty for all)", staff_list)

if selected_staff:
    all_events = all_events[all_events["who"].isin(selected_staff)]

# --- KPIs
k1, k2, k3, k4 = st.columns(4)
total_all = len(all_events)
total_cases = (all_events["source"] == "Klinik Cases").sum()
total_doc = (all_events["source"] == "Docman Tasks").sum()
total_calls = (all_events["source"] == "Calls (Answered)").sum()

k1.metric("Total tasks", total_all)
k2.metric("Klinik cases closed", int(total_cases))
k3.metric("Docman completed", int(total_doc))
k4.metric("Calls answered/callbacks", int(total_calls))

# --- Prepare hourly-by-day matrix
df_grid = all_events.copy()
df_grid["date"] = df_grid["when"].dt.date
df_grid["weekday"] = df_grid["when"].dt.strftime("%a %d %b")
df_grid["hour"] = df_grid["when"].dt.hour

pivot = df_grid.pivot_table(index="weekday", columns="hour", values="who", aggfunc="count", fill_value=0)
# Order days chronologically within the range
ordered_days = sorted(df_grid["weekday"].unique().tolist(), key=lambda s: datetime.strptime(s, "%a %d %b"))
pivot = pivot.reindex(ordered_days)

st.markdown("### ⏱️ Tasks completed by hour each day")
fig = px.imshow(pivot.values,
                labels=dict(x="Hour of day", y="Day", color="Tasks"),
                x=pivot.columns,
                y=pivot.index,
                aspect="auto",
                text_auto=True)
st.plotly_chart(fig, use_container_width=True)

# --- Breakdown by staff
st.markdown("### 👤 Staff breakdown")
by_staff = all_events.groupby(["who", "source"]).size().unstack(fill_value=0)
by_staff["Total"] = by_staff.sum(axis=1)
by_staff = by_staff.sort_values("Total", ascending=False)

st.dataframe(by_staff, use_container_width=True)

# --- Timeline chart
st.markdown("### 📈 Timeline — tasks per hour (stacked by source)")
timeline = df_grid.groupby([pd.Grouper(key="when", freq="H"), "source"]).size().reset_index(name="count")
fig2 = px.bar(timeline, x="when", y="count", color="source", barmode="stack")
fig2.update_layout(xaxis_title="Time", yaxis_title="Tasks per hour")
st.plotly_chart(fig2, use_container_width=True)

st.success("Done! Use the file uploaders and date pickers at the top to refresh the dashboard with new reports and dates.")
