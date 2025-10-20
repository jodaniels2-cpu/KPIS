
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

# -----------------------------
# Page config & Theming
# -----------------------------
st.set_page_config(
    page_title="A-Team KPI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def parse_csv(file):
    if file is None:
        return None
    # Try a few common encodings/separators
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
    return pd.read_csv(file)  # last resort

def parse_dt(date_series, time_series=None):
    if time_series is None:
        return pd.to_datetime(date_series, errors="coerce", dayfirst=True)
    combined = date_series.astype(str).str.strip() + " " + time_series.astype(str).str.strip()
    return pd.to_datetime(combined, errors="coerce", dayfirst=True)

def clean_staff(val: str):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if "@" in s:
        return s.lower()
    return " ".join(w.capitalize() for w in s.split() if w)

def kpi_card(label: str, value: int, help_text: str = ""):
    st.metric(label, f"{int(value):,}", help=None, help=help_text if help_text else None)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
# 📊 A‑Team KPI Dashboard
Upload your weekly **Klinik**, **Docman10**, and **Calls** CSVs, pick the **date range**, and drill down by **staff**.
""")

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("Controls")
    col1, col2 = st.columns(2)
    today = date.today()
    default_start = today - timedelta(days=today.weekday())  # Mon
    default_end = default_start + timedelta(days=4)  # Fri

    start = st.date_input("Start date", value=default_start, key="start_date")
    end = st.date_input("End date", value=default_end, key="end_date")
    if end < start:
        st.error("End date must be on or after start date")
        st.stop()

    st.divider()
    st.subheader("Upload files")
    case_file = st.file_uploader("Klinik Case Counts (CSV)", type=["csv"], key="case")
    docman_file = st.file_uploader("Docman10 Tasks (CSV)", type=["csv"], key="docman")
    calls_file = st.file_uploader("Telephone Calls Export (CSV)", type=["csv"], key="calls")

    st.caption("Tip: Column mapping is configurable below if your headers differ.")

# -----------------------------
# Load Files
# -----------------------------
events = []
case_df = parse_csv(case_file) if case_file else None
doc_df = parse_csv(docman_file) if docman_file else None
calls_df = parse_csv(calls_file) if calls_file else None

# -----------------------------
# Mapping widgets
# -----------------------------
with st.expander("🗂️ Column mapping (adjust only if needed)", expanded=True):
    if case_df is not None:
        st.markdown("**Klinik Cases**")
        c1, c2, c3 = st.columns(3)
        case_staff_col = c1.selectbox("Staff (e.g., last_archived_by)", options=case_df.columns, index=next((i for i, c in enumerate(case_df.columns.str.lower()) if c == "last_archived_by"), 0))
        case_date_col = c2.selectbox("Date (e.g., last_archived_date)", options=case_df.columns, index=next((i for i, c in enumerate(case_df.columns.str.lower()) if c == "last_archived_date"), 0))
        case_time_col = c3.selectbox("Time (e.g., last_archived_time)", options=case_df.columns, index=next((i for i, c in enumerate(case_df.columns.str.lower()) if c == "last_archived_time"), 0))

        case_dt = parse_dt(case_df[case_date_col], case_df[case_time_col])
        case_staff = case_df[case_staff_col].map(clean_staff)
        case_events = pd.DataFrame({"when": case_dt, "who": case_staff, "source": "Klinik Cases"})
        events.append(case_events)

    if doc_df is not None:
        st.markdown("**Docman10 Tasks**")
        d1, d2 = st.columns(2)
        doc_staff_col = d1.selectbox("Completed User", options=doc_df.columns, index=next((i for i, c in enumerate(doc_df.columns.str.lower()) if c == "completed user"), 0))
        doc_dt_col = d2.selectbox("Date & Time of Event", options=doc_df.columns, index=next((i for i, c in enumerate(doc_df.columns.str.lower()) if c == "date and time of event"), 0))

        doc_dt = parse_dt(doc_df[doc_dt_col], None)
        doc_staff = doc_df[doc_staff_col].map(clean_staff)
        doc_events = pd.DataFrame({"when": doc_dt, "who": doc_staff, "source": "Docman Tasks"})
        events.append(doc_events)

    if calls_df is not None:
        st.markdown("**Telephone Calls**")
        e1, e2, e3, e4, e5 = st.columns(5)
        call_user_col = e1.selectbox("Inbound staff: User Name", options=calls_df.columns, index=next((i for i, c in enumerate(calls_df.columns.str.lower()) if c == "user name"), 0))
        call_caller_col = e2.selectbox("Callback staff: Caller Name", options=calls_df.columns, index=next((i for i, c in enumerate(calls_df.columns.str.lower()) if c == "caller name"), 0))
        call_outcome_col = e3.selectbox("Outcome", options=calls_df.columns, index=next((i for i, c in enumerate(calls_df.columns.str.lower()) if c == "outcome"), 0))
        call_direction_col = e4.selectbox("Direction", options=calls_df.columns, index=next((i for i, c in enumerate(calls_df.columns.str.lower()) if c == "direction"), 0))
        call_start_col = e5.selectbox("Start Time", options=calls_df.columns, index=next((i for i, c in enumerate(calls_df.columns.str.lower()) if c == "start time"), 0))

        call_dt = parse_dt(calls_df[call_start_col], None)
        dir_lower = calls_df[call_direction_col].astype(str).str.lower()
        out_lower = calls_df[call_outcome_col].astype(str).str.lower()
        inbound_mask = dir_lower.str.contains("inbound")
        answered_mask = out_lower.str.contains("answer|connect|complete|handled")

        who = np.where(inbound_mask, calls_df[call_user_col].astype(str), calls_df[call_caller_col].astype(str))
        who = pd.Series(who).map(clean_staff)
        call_events = pd.DataFrame({"when": call_dt, "who": who, "source": "Calls (Answered)"})
        try:
            call_events = call_events[answered_mask.values]
        except Exception:
            pass
        events.append(call_events)

# -----------------------------
# Combine & filter
# -----------------------------
if not events:
    st.info("Upload at least one file to see the dashboard.")
    st.stop()

all_events = pd.concat(events, ignore_index=True)
all_events = all_events.dropna(subset=["when", "who"])
start_dt = datetime.combine(start, datetime.min.time())
end_dt = datetime.combine(end, datetime.max.time())
mask = (all_events["when"] >= pd.Timestamp(start_dt)) & (all_events["when"] <= pd.Timestamp(end_dt))
all_events = all_events.loc[mask].copy()

if all_events.empty:
    st.warning("No events found in the selected date range.")
    st.stop()

# Filters
staff_list = sorted(all_events["who"].dropna().unique())
st.subheader("Filters")
fc1, fc2, fc3 = st.columns([2, 2, 1])
with fc1:
    selected_staff = st.multiselect("Staff (leave empty for all)", staff_list)
with fc2:
    source_opts = sorted(all_events["source"].unique().tolist())
    selected_sources = st.multiselect("Sources", source_opts, default=source_opts)
with fc3:
    weekdays_only = st.toggle("Weekdays only", value=True)

df = all_events.copy()
if selected_staff:
    df = df[df["who"].isin(selected_staff)]
if selected_sources:
    df = df[df["source"].isin(selected_sources)]
if weekdays_only:
    df = df[df["when"].dt.weekday < 5]

# Derive time columns
df["day"] = df["when"].dt.strftime("%a %d %b")
df["hour"] = df["when"].dt.hour
ordered_days = sorted(df["day"].unique().tolist(), key=lambda s: datetime.strptime(s, "%a %d %b"))

# -----------------------------
# KPI Row (cards)
# -----------------------------
st.markdown("### KPI Overview")
k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card("Total tasks", len(df))
with k2:
    kpi_card("Klinik cases closed", int((df["source"] == "Klinik Cases").sum()))
with k3:
    kpi_card("Docman completed", int((df["source"] == "Docman Tasks").sum()))
with k4:
    kpi_card("Calls answered / callbacks", int((df["source"] == "Calls (Answered)").sum()))

st.divider()

# -----------------------------
# Tabs for layout
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "👤 Staff", "📅 Daily", "📥 Data"])

# -------- Overview
with tab1:
    st.markdown("#### Tasks completed by hour each day (heatmap)")
    pivot = df.pivot_table(index="day", columns="hour", values="who", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(ordered_days)
    fig = px.imshow(
        pivot.values,
        labels=dict(x="Hour of day", y="Day", color="Tasks"),
        x=pivot.columns,
        y=pivot.index,
        aspect="auto",
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Hourly timeline (stacked by source)")
    tl = df.groupby([pd.Grouper(key="when", freq="H"), "source"]).size().reset_index(name="count")
    fig2 = px.bar(tl, x="when", y="count", color="source", barmode="stack")
    fig2.update_layout(xaxis_title="Time", yaxis_title="Tasks per hour")
    st.plotly_chart(fig2, use_container_width=True)

# -------- Staff
with tab2:
    st.markdown("#### Staff totals by source")
    by_staff = df.groupby(["who", "source"]).size().unstack(fill_value=0)
    by_staff["Total"] = by_staff.sum(axis=1)
    by_staff = by_staff.sort_values("Total", ascending=False)
    st.dataframe(by_staff, use_container_width=True)

    st.markdown("#### Top staff this week")
    top_staff = by_staff["Total"].reset_index().sort_values("Total", ascending=False).head(10)
    fig3 = px.bar(top_staff, x="who", y="Total")
    fig3.update_layout(xaxis_title="Staff", yaxis_title="Total tasks")
    st.plotly_chart(fig3, use_container_width=True)

# -------- Daily
with tab3:
    st.markdown("#### Daily totals by source")
    day_src = df.groupby(["day", "source"]).size().reset_index(name="count")
    day_src["day_order"] = day_src["day"].apply(lambda s: ordered_days.index(s))
    day_src = day_src.sort_values("day_order")
    fig4 = px.bar(day_src, x="day", y="count", color="source", barmode="stack")
    fig4.update_layout(xaxis_title="Day", yaxis_title="Tasks")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Daily totals by staff (top 12)")
    day_staff = df.groupby(["day", "who"]).size().reset_index(name="count")
    # keep top staff overall to reduce clutter
    top12 = df["who"].value_counts().head(12).index.tolist()
    day_staff = day_staff[day_staff["who"].isin(top12)]
    fig5 = px.bar(day_staff, x="day", y="count", color="who", barmode="stack")
    fig5.update_layout(xaxis_title="Day", yaxis_title="Tasks")
    st.plotly_chart(fig5, use_container_width=True)

# -------- Data
with tab4:
    st.markdown("#### Underlying events")
    st.dataframe(df.sort_values("when"), use_container_width=True, height=420)
    csv = df.sort_values("when").to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data (CSV)", data=csv, file_name="kpi_events_filtered.csv", mime="text/csv")
