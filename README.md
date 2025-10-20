
# A‑Team KPI Dashboard (Streamlit)

This is a zero‑install, browser‑based dashboard for:
- Klinik case closures
- Docman10 completed tasks
- Telephone calls (inbound + callbacks)

## Quick deploy on Streamlit Cloud
1. Create a **public** GitHub repo (e.g. `a-team-kpi-dashboard`).
2. Upload the three files in this folder:
   - `kpi_dashboard_app.py`
   - `requirements_kpi_dashboard.txt`
   - `README.md` (optional)
3. Go to https://streamlit.io/cloud → **Deploy a public app from GitHub**
   - Repo: `yourusername/a-team-kpi-dashboard`
   - Branch: `main`
   - File path: `kpi_dashboard_app.py`
4. Click **Deploy**. Share the URL with your team.

## Using the app
- Upload the three CSVs (Klinik, Docman10, Calls).
- Pick your date range (e.g. 13–17 Oct 2025).
- (Optional) Filter by staff member.
- See KPIs, an hourly heatmap per day, staff breakdown, and an hourly timeline.

## Notes
- Column mapping is built‑in, so exact column names are not required.
- Calls: inbound uses "User Name"; callbacks/outbound use "Caller Name".
- Answered/connected/handled calls are counted for calls KPI/plots.
