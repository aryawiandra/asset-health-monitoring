"""
Asset Health Monitoring Dashboard
Streamlit app — run with: streamlit run dashboard/app.py
"""
import os, json, joblib, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Asset Health Monitor",
    page_icon  = "🔧",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
.status-critical { color: #E24B4A; font-weight: 600; }
.status-warning  { color: #BA7517; font-weight: 600; }
.status-healthy  { color: #1D9E75; font-weight: 600; }
div[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* Forecast risk badge */
.risk-critical { background:#E24B4A; color:white; padding:2px 8px; border-radius:4px; font-size:0.8rem; font-weight:600; }
.risk-high     { background:#C0392B; color:white; padding:2px 8px; border-radius:4px; font-size:0.8rem; font-weight:600; }
.risk-medium   { background:#BA7517; color:white; padding:2px 8px; border-radius:4px; font-size:0.8rem; font-weight:600; }
.risk-low      { background:#1D9E75; color:white; padding:2px 8px; border-radius:4px; font-size:0.8rem; font-weight:600; }

/* Forecast card styling */
.forecast-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ──────────────────────────────────────────────────────────────
SENSOR_COLS   = ["volt", "rotate", "pressure", "vibration"]
SENSOR_UNITS  = {"volt":"V", "rotate":"rpm", "pressure":"psi", "vibration":"mm/s"}
SENSOR_COLORS = {"volt":"#185FA5","rotate":"#1D9E75","pressure":"#BA7517","vibration":"#993C1D"}

STATUS_COLOR  = {"Critical":"#E24B4A", "Warning":"#BA7517", "Healthy":"#1D9E75"}
STATUS_ICON   = {"Critical":"🔴", "Warning":"🟡", "Healthy":"🟢"}

RISK_COLOR    = {"Critical":"#E24B4A", "High":"#C0392B", "Medium":"#BA7517", "Low":"#1D9E75"}
RISK_ICON     = {"Critical":"🔴", "High":"🟠", "Medium":"🟡", "Low":"🟢"}

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# ─── Load data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(PROC_DIR, "scored.parquet")
    if not os.path.exists(path):
        st.error(f"File not found: {path}\nJalankan notebook 02 dan 03 terlebih dahulu.")
        st.stop()
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

df = load_data()

# ─── Forecast Engine ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_all_forecasts(df_hash_key, lookback_days: int = 14):
    """
    Compute forecast for every machine in the dataset.

    Two-layer approach:
      1. Linear regression on health_score over last `lookback_days` → extrapolate
      2. Exponential smoothing on daily anomaly rate → project forward

    Returns a DataFrame with one row per machine:
        machineID, current_score, forecast_72h, forecast_7d,
        anom_rate_current, anom_rate_7d, days_to_critical,
        trend_slope, risk_level, confidence_72h
    """
    results = []
    cutoff  = df["datetime"].max() - pd.Timedelta(days=lookback_days)

    for mid, mdf in df.groupby("machineID"):
        window = mdf[mdf["datetime"] >= cutoff].copy()
        if len(window) < 12:          # need at least 12 hours of data
            window = mdf.tail(168)    # fallback: last 7 days

        # ── Health score trend (linear regression) ──────────────────────────
        t = np.arange(len(window)).reshape(-1, 1)
        y = window["health_score"].values

        reg = LinearRegression().fit(t, y)
        slope = reg.coef_[0]            # score change per hour

        # Residual std → confidence
        y_pred = reg.predict(t)
        residual_std = np.std(y - y_pred)

        n_hours = len(window)
        forecast_72h_val = float(np.clip(reg.predict([[n_hours + 72]])[0], 0, 100))
        forecast_7d_val  = float(np.clip(reg.predict([[n_hours + 168]])[0], 0, 100))

        ci_72h = float(residual_std * 1.0)   # ±1 std
        ci_7d  = float(residual_std * 1.5)

        # ── Anomaly rate (exponential smoothing) ────────────────────────────
        daily = (window.groupby(window["datetime"].dt.floor("D"))
                 .agg(n_anom=("if_anomaly_01", "sum"),
                      n_total=("if_anomaly_01", "count"))
                 .assign(anom_rate=lambda x: x["n_anom"] / x["n_total"].clip(lower=1)))

        alpha = 0.3   # smoothing factor
        smoothed = daily["anom_rate"].ewm(alpha=alpha, adjust=False).mean()
        current_anom_rate = float(smoothed.iloc[-1]) if len(smoothed) else 0.0

        # Project anomaly rate (exponential decay/growth continues)
        last_rate = smoothed.iloc[-1] if len(smoothed) else 0.0
        rate_slope = (smoothed.diff().mean()) if len(smoothed) > 1 else 0.0
        projected_anom_rate_7d = float(np.clip(last_rate + rate_slope * 7, 0, 1))

        # ── Days to critical ────────────────────────────────────────────────
        current_score = float(window["health_score"].iloc[-1])
        critical_thresh = 40   # default; will be overridden by slider where used
        if slope < 0 and current_score > critical_thresh:
            hours_to_crit = (critical_thresh - current_score) / slope
            days_to_crit  = float(np.clip(hours_to_crit / 24, 0, 365))
        else:
            days_to_crit  = 999.0   # not trending toward critical

        # ── Risk level ──────────────────────────────────────────────────────
        def compute_risk(f72, f7d, anom_rate):
            score_min = min(f72, f7d, current_score)
            if score_min < 40 or anom_rate > 0.5:
                return "Critical"
            elif score_min < 55 or anom_rate > 0.35:
                return "High"
            elif score_min < 70 or anom_rate > 0.20:
                return "Medium"
            else:
                return "Low"

        risk = compute_risk(forecast_72h_val, forecast_7d_val, current_anom_rate)

        results.append({
            "machineID"          : mid,
            "current_score"      : current_score,
            "forecast_72h"       : forecast_72h_val,
            "forecast_7d"        : forecast_7d_val,
            "ci_72h"             : ci_72h,
            "ci_7d"              : ci_7d,
            "trend_slope"        : slope * 24,  # per day
            "anom_rate_current"  : current_anom_rate,
            "anom_rate_7d"       : projected_anom_rate_7d,
            "days_to_critical"   : days_to_crit,
            "risk_level"         : risk,
        })

    return pd.DataFrame(results)


@st.cache_data(show_spinner=False)
def compute_machine_forecast_series(machine_id, lookback_days: int = 14):
    """
    Returns the historical window + projected future time series for a single machine.
    Used for the forecast chart in Machine Detail.
    """
    mdf    = df[df["machineID"] == machine_id].copy()
    cutoff = df["datetime"].max() - pd.Timedelta(days=lookback_days)
    window = mdf[mdf["datetime"] >= cutoff].copy()
    if len(window) < 12:
        window = mdf.tail(168)

    t = np.arange(len(window)).reshape(-1, 1)
    y = window["health_score"].values
    reg = LinearRegression().fit(t, y)
    residual_std = np.std(y - reg.predict(t))

    # Build future timestamps (hourly)
    last_dt   = window["datetime"].iloc[-1]
    future_72 = [last_dt + pd.Timedelta(hours=h) for h in range(1, 73)]
    future_7d = [last_dt + pd.Timedelta(hours=h) for h in range(1, 169)]

    n = len(window)
    fut_72_vals = reg.predict(np.arange(n, n + 72).reshape(-1, 1)).clip(0, 100)
    fut_7d_vals = reg.predict(np.arange(n, n + 168).reshape(-1, 1)).clip(0, 100)

    return {
        "hist_dt"   : window["datetime"].tolist(),
        "hist_score": window["health_score"].tolist(),
        "fut_dt_7d" : future_7d,
        "fut_score_7d": fut_7d_vals.flatten().tolist(),
        "ci"        : residual_std,
    }


# Compute forecasts once per session (cache busted by dataset max date)
_forecast_key = str(df["datetime"].max())
forecast_df = compute_all_forecasts(_forecast_key, lookback_days=14)

# ─── Sidebar filters ─────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/Pertamina_logo.svg/200px-Pertamina_logo.svg.png",
                 width=140)
st.sidebar.markdown("## Asset Health Monitor")
st.sidebar.markdown("*Pertamina EDM — Mini Project KP*")
st.sidebar.divider()

all_models = sorted(df["model"].unique())
sel_models = st.sidebar.multiselect("Machine model", all_models, default=all_models)

date_min = df["datetime"].min().date()
date_max = df["datetime"].max().date()
sel_dates = st.sidebar.date_input("Date range", value=[date_min, date_max])

st.sidebar.divider()
st.sidebar.markdown("**Health thresholds**")
thresh_critical = st.sidebar.slider("Critical below", 0, 60, 40)
thresh_warning  = st.sidebar.slider("Warning below",  thresh_critical, 90, 70)

# ─── Filter ──────────────────────────────────────────────────────────────────
if len(sel_dates) == 2:
    d0, d1 = pd.Timestamp(sel_dates[0]), pd.Timestamp(sel_dates[1])
else:
    d0, d1 = pd.Timestamp(date_min), pd.Timestamp(date_max)

df_filt = df[
    (df["model"].isin(sel_models)) &
    (df["datetime"] >= d0) &
    (df["datetime"] <= d1)
].copy()

# Re-apply threshold labels based on slider
def status(score):
    if score < thresh_critical: return "Critical"
    elif score < thresh_warning: return "Warning"
    return "Healthy"
df_filt["health_status"] = df_filt["health_score"].apply(status)

# Latest reading per machine
latest = df_filt.groupby("machineID").last().reset_index()

# Merge forecast data into latest (for KPIs)
latest = latest.merge(forecast_df[["machineID","forecast_72h","forecast_7d",
                                    "risk_level","days_to_critical","trend_slope"]],
                       on="machineID", how="left")

# ─── Page navigation ─────────────────────────────────────────────────────────
page = st.sidebar.radio("Navigation", [
    "Fleet Overview",
    "Machine Detail",
    "Anomaly Timeline",
    "🔮 Anomaly Forecast",
])

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1: FLEET OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "Fleet Overview":
    st.title("🔧 Asset Health — Fleet Overview")
    st.caption(f"Data: {d0.date()} to {d1.date()} | "
               f"{df_filt['machineID'].nunique()} machines | "
               f"{sel_models} models")
    st.divider()

    # KPI row 1 — current status
    n_crit = (latest["health_status"] == "Critical").sum()
    n_warn = (latest["health_status"] == "Warning").sum()
    n_ok   = (latest["health_status"] == "Healthy").sum()
    total  = len(latest)
    anom   = df_filt["if_anomaly_01"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Machines",  total)
    c2.metric("🔴 Critical",     n_crit,  delta=None)
    c3.metric("🟡 Warning",      n_warn,  delta=None)
    c4.metric("🟢 Healthy",      n_ok,    delta=None)
    c5.metric("⚠️ Anomaly Events", f"{anom:,}", delta=None)

    # KPI row 2 — forecast
    fcast_sub = forecast_df[forecast_df["machineID"].isin(latest["machineID"])]
    n_risk_72h = (fcast_sub["risk_level"].isin(["Critical","High"])).sum()
    n_risk_7d  = (fcast_sub["risk_level"] == "Critical").sum()
    n_declining = (fcast_sub["trend_slope"] < -2).sum()     # losing >2 pts/day
    avg_forecast_7d = fcast_sub["forecast_7d"].mean()

    st.markdown("**📡 Forecast Outlook (based on recent trends)**")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("🚨 High-Risk (72h)",
              n_risk_72h,
              delta=f"{n_risk_72h/total*100:.0f}% of fleet",
              delta_color="inverse")
    f2.metric("📅 Critical Risk (7d)",
              n_risk_7d,
              delta=f"{n_risk_7d/total*100:.0f}% of fleet",
              delta_color="inverse")
    f3.metric("📉 Machines Declining",
              n_declining,
              delta="trend slope < -2 pts/day",
              delta_color="inverse")
    f4.metric("🏥 Avg Fleet Score (7d)",
              f"{avg_forecast_7d:.1f}",
              delta=f"{avg_forecast_7d - latest['health_score'].mean():.1f} vs now",
              delta_color="normal")

    st.divider()

    # Fleet health score distribution
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Fleet Health Score Distribution")
        fig = go.Figure()
        for status_val in ["Healthy","Warning","Critical"]:
            sub = latest[latest["health_status"]==status_val]
            fig.add_trace(go.Bar(
                x=sub["machineID"].astype(str),
                y=sub["health_score"],
                name=status_val,
                marker_color=STATUS_COLOR[status_val],
                opacity=0.85,
                hovertemplate="Machine %{x}<br>Health: %{y:.1f}<extra></extra>",
            ))
        fig.add_hline(y=thresh_warning,  line_dash="dash", line_color="#1D9E75",
                      annotation_text="Warning threshold")
        fig.add_hline(y=thresh_critical, line_dash="dash", line_color="#E24B4A",
                      annotation_text="Critical threshold")
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Machine ID",
            yaxis_title="Health Score",
            yaxis_range=[0, 105],
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0,r=0,t=30,b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Status Breakdown")
        fig_pie = go.Figure(go.Pie(
            labels=["Healthy","Warning","Critical"],
            values=[n_ok, n_warn, n_crit],
            marker_colors=["#1D9E75","#BA7517","#E24B4A"],
            hole=0.45,
            textinfo="percent+label",
        ))
        fig_pie.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0),
                               showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown(f"""
        | Status | Count | % |
        |--------|-------|---|
        | 🔴 Critical | {n_crit} | {n_crit/total*100:.0f}% |
        | 🟡 Warning  | {n_warn} | {n_warn/total*100:.0f}% |
        | 🟢 Healthy  | {n_ok}   | {n_ok/total*100:.0f}%  |
        """)

    st.divider()

    # Machine table
    st.subheader("Machine Status Table")
    table_data = latest[["machineID","model","age","health_score","health_status",
                          "total_error_count24h","days_since_maint",
                          "forecast_72h","forecast_7d","risk_level"]].copy()
    table_data = table_data.sort_values("health_score")
    table_data["health_score"]    = table_data["health_score"].round(1)
    table_data["days_since_maint"]= table_data["days_since_maint"].round(1)
    table_data["forecast_72h"]    = table_data["forecast_72h"].round(1)
    table_data["forecast_7d"]     = table_data["forecast_7d"].round(1)
    table_data["Status"] = table_data["health_status"].map(
        lambda s: f"{STATUS_ICON[s]} {s}")
    table_data["Risk (7d)"] = table_data["risk_level"].map(
        lambda r: f"{RISK_ICON.get(r,'❓')} {r}")

    st.dataframe(
        table_data[["machineID","model","age","health_score","Status",
                    "total_error_count24h","days_since_maint",
                    "forecast_72h","forecast_7d","Risk (7d)"]].rename(columns={
            "machineID":"Machine ID","model":"Model","age":"Age (yr)",
            "health_score":"Health Score","total_error_count24h":"Errors 24h",
            "days_since_maint":"Days Since Maint.",
            "forecast_72h":"Forecast 72h","forecast_7d":"Forecast 7d",
        }),
        height=400,
        use_container_width=True,
    )

    csv = table_data.to_csv(index=False).encode("utf-8")
    st.download_button("Export to CSV", csv, "fleet_status.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2: MACHINE DETAIL
# ════════════════════════════════════════════════════════════════════════════
elif page == "Machine Detail":
    st.title("🔍 Machine Detail")

    machine_ids = sorted(df_filt["machineID"].unique())
    sel_machine = st.selectbox("Select Machine ID", machine_ids)

    mdf      = df_filt[df_filt["machineID"] == sel_machine].copy()
    m_latest = mdf.iloc[-1]

    # Machine KPIs
    st.divider()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Machine", sel_machine)
    k2.metric("Model", m_latest["model"])
    k3.metric("Age", f"{m_latest['age']} yr")
    status_now = m_latest["health_status"]
    k4.metric("Status", f"{STATUS_ICON[status_now]} {status_now}")
    k5.metric("Health Score", f"{m_latest['health_score']:.1f}")

    # Gauge chart
    st.subheader("Current Health Score")
    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = m_latest["health_score"],
        delta = {"reference": 70, "increasing": {"color":"#1D9E75"}},
        gauge = {
            "axis"  : {"range":[0,100], "tickwidth":1},
            "bar"   : {"color": STATUS_COLOR[status_now]},
            "steps" : [
                {"range":[0,  thresh_critical], "color":"#FAE0DF"},
                {"range":[thresh_critical, thresh_warning], "color":"#FDF0D5"},
                {"range":[thresh_warning,  100], "color":"#D7F4EA"},
            ],
            "threshold": {
                "line"  : {"color":"black","width":3},
                "thickness": 0.75,
                "value" : thresh_warning,
            },
        },
        title = {"text": f"Machine {sel_machine} — Health Score"},
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=40,r=40,t=40,b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # Sensor time series
    st.subheader("Sensor Readings Over Time")
    fig_sensors = make_subplots(rows=2, cols=2, shared_xaxes=True,
                                subplot_titles=[s.capitalize() for s in SENSOR_COLS],
                                vertical_spacing=0.1)
    positions = [(1,1),(1,2),(2,1),(2,2)]

    for (r,c), sensor in zip(positions, SENSOR_COLS):
        col  = SENSOR_COLORS[sensor]
        unit = SENSOR_UNITS[sensor]
        fig_sensors.add_trace(
            go.Scatter(x=mdf["datetime"], y=mdf[sensor],
                       name=sensor, line=dict(color=col, width=1.0),
                       opacity=0.8,
                       hovertemplate=f"%{{x}}<br>{sensor}: %{{y:.2f}} {unit}<extra></extra>"),
            row=r, col=c)
        fig_sensors.add_trace(
            go.Scatter(x=mdf["datetime"], y=mdf[f"{sensor}_mean24h"],
                       name=f"{sensor} 24h mean", line=dict(color="black", width=1.5, dash="dash"),
                       opacity=0.5, showlegend=(r==1 and c==1)),
            row=r, col=c)
        anom = mdf[mdf["if_anomaly_01"]==1]
        fig_sensors.add_trace(
            go.Scatter(x=anom["datetime"], y=anom[sensor],
                       mode="markers", name="Anomaly",
                       marker=dict(color="#E24B4A", size=4, symbol="circle"),
                       showlegend=(r==1 and c==1)),
            row=r, col=c)

    fig_sensors.update_layout(height=420, margin=dict(l=0,r=0,t=40,b=0),
                               hovermode="x unified")
    st.plotly_chart(fig_sensors, use_container_width=True)

    # Health score over time
    st.subheader("Health Score Timeline")
    fig_hs = go.Figure()
    fig_hs.add_trace(go.Scatter(
        x=mdf["datetime"], y=mdf["health_score"],
        fill="tozeroy",
        fillcolor="rgba(29,158,117,0.15)",
        line=dict(color="#1D9E75", width=1.5),
        name="Health Score",
    ))
    anom = mdf[mdf["if_anomaly_01"]==1]
    fig_hs.add_trace(go.Scatter(
        x=anom["datetime"], y=anom["health_score"],
        mode="markers", name="Anomaly",
        marker=dict(color="#E24B4A", size=5),
    ))
    fig_hs.add_hline(y=thresh_warning,  line_dash="dash", line_color="#1D9E75",
                     annotation_text="Warning")
    fig_hs.add_hline(y=thresh_critical, line_dash="dash", line_color="#E24B4A",
                     annotation_text="Critical")
    fig_hs.update_layout(
        height=220, yaxis_range=[0,105], yaxis_title="Score",
        margin=dict(l=0,r=0,t=20,b=0), hovermode="x unified",
    )
    st.plotly_chart(fig_hs, use_container_width=True)

    # ── FORECAST SECTION ────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔮 Anomaly Forecast")
    st.caption("Proyeksi health score berdasarkan tren 14 hari terakhir. "
               "Shaded area = confidence interval (±1 std residual).")

    # Forecast summary cards
    m_fcast = forecast_df[forecast_df["machineID"] == sel_machine].iloc[0]
    f72  = m_fcast["forecast_72h"]
    f7d  = m_fcast["forecast_7d"]
    ci72 = m_fcast["ci_72h"]
    ci7d = m_fcast["ci_7d"]
    dtc  = m_fcast["days_to_critical"]
    risk = m_fcast["risk_level"]
    slope_per_day = m_fcast["trend_slope"]

    def score_to_status(s):
        if s < thresh_critical: return "Critical", STATUS_COLOR["Critical"]
        elif s < thresh_warning: return "Warning",  STATUS_COLOR["Warning"]
        return "Healthy", STATUS_COLOR["Healthy"]

    _, col_a, col_b, col_c, col_d = st.columns([0.05, 1, 1, 1, 1])

    with col_a:
        stat72, col72 = score_to_status(f72)
        st.metric("📊 Score (72h)",
                  f"{f72:.1f}",
                  delta=f"{f72 - m_latest['health_score']:.1f} vs now",
                  delta_color="normal")
        st.caption(f"CI: ±{ci72:.1f} | Status: {STATUS_ICON[stat72]} {stat72}")

    with col_b:
        stat7d, col7d = score_to_status(f7d)
        st.metric("📅 Score (7d)",
                  f"{f7d:.1f}",
                  delta=f"{f7d - m_latest['health_score']:.1f} vs now",
                  delta_color="normal")
        st.caption(f"CI: ±{ci7d:.1f} | Status: {STATUS_ICON[stat7d]} {stat7d}")

    with col_c:
        trend_str = f"{slope_per_day:+.2f} pts/day"
        trend_ico = "📉" if slope_per_day < 0 else "📈"
        st.metric(f"{trend_ico} Trend",
                  trend_str,
                  delta="slope (linear fit)",
                  delta_color="off")

    with col_d:
        if dtc < 365:
            st.metric("⏰ Days to Critical",
                      f"{dtc:.1f} days",
                      delta="if trend continues",
                      delta_color="inverse")
        else:
            st.metric("⏰ Days to Critical",
                      "N/A",
                      delta="not trending critical",
                      delta_color="off")

    # Forecast chart
    with st.spinner("Computing forecast series..."):
        fc_series = compute_machine_forecast_series(sel_machine, lookback_days=14)

    fig_fc = go.Figure()

    # Historical
    fig_fc.add_trace(go.Scatter(
        x=fc_series["hist_dt"], y=fc_series["hist_score"],
        name="Historical (14d)", line=dict(color="#185FA5", width=2),
        hovertemplate="%{x}<br>Score: %{y:.1f}<extra>Historical</extra>",
    ))

    # Forecast line
    fig_fc.add_trace(go.Scatter(
        x=fc_series["fut_dt_7d"], y=fc_series["fut_score_7d"],
        name="Forecast (7d)", line=dict(color="#BA7517", width=2, dash="dot"),
        hovertemplate="%{x}<br>Forecast: %{y:.1f}<extra>Projected</extra>",
    ))

    # Confidence interval (upper + lower as filled band)
    ci = fc_series["ci"]
    upper = [v + ci * 1.5 for v in fc_series["fut_score_7d"]]
    lower = [max(0, v - ci * 1.5) for v in fc_series["fut_score_7d"]]

    fig_fc.add_trace(go.Scatter(
        x=fc_series["fut_dt_7d"] + fc_series["fut_dt_7d"][::-1],
        y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(186,117,23,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Interval",
        hoverinfo="skip",
    ))

    # Threshold lines
    fig_fc.add_hline(y=thresh_warning,  line_dash="dash", line_color="#1D9E75",
                     annotation_text="Warning")
    fig_fc.add_hline(y=thresh_critical, line_dash="dash", line_color="#E24B4A",
                     annotation_text="Critical")

    # Vertical "now" line — convert to string to avoid Timestamp arithmetic TypeError
    now_dt = fc_series["hist_dt"][-1]
    now_dt_str = pd.Timestamp(now_dt).isoformat()
    fig_fc.add_vline(x=now_dt_str, line_dash="solid", line_color="rgba(255,255,255,0.3)",
                     annotation_text="Now")

    fig_fc.update_layout(
        height=320,
        yaxis_range=[0, 105],
        yaxis_title="Health Score",
        xaxis_title="Datetime",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_fc, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3: ANOMALY TIMELINE
# ════════════════════════════════════════════════════════════════════════════
elif page == "Anomaly Timeline":
    st.title("⚠️ Anomaly Timeline")
    st.caption("Scatter plot seluruh anomali yang terdeteksi dalam fleet.")

    df_anom = df_filt[df_filt["if_anomaly_01"]==1].copy()
    df_anom["date"] = df_anom["datetime"].dt.floor("D")
    daily_anom = (df_anom.groupby(["machineID","date"])
                  .agg(n_anomalies=("if_anomaly_01","sum"),
                       avg_health=("health_score","mean"))
                  .reset_index())
    daily_anom["status"] = daily_anom["avg_health"].apply(status)

    # Scatter heatmap
    st.subheader("Anomaly Events — Fleet Heatmap (Daily)")
    fig = px.scatter(
        daily_anom,
        x="date", y="machineID",
        size="n_anomalies",
        color="avg_health",
        color_continuous_scale=["#E24B4A","#BA7517","#1D9E75"],
        range_color=[0,100],
        labels={"date":"Date","machineID":"Machine ID",
                "n_anomalies":"Anomalies","avg_health":"Avg Health Score"},
        hover_data={"date":True,"machineID":True,"n_anomalies":True,"avg_health":":.1f"},
        height=600,
    )
    fig.update_traces(marker=dict(opacity=0.75))
    fig.update_layout(
        coloraxis_colorbar=dict(title="Health Score"),
        margin=dict(l=0,r=0,t=30,b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Anomaly Count Over Time — Fleet Total")

    fleet_daily = (df_filt.groupby(df_filt["datetime"].dt.floor("D"))
                   .agg(n_anomalies=("if_anomaly_01","sum"),
                        avg_health=("health_score","mean"))
                   .reset_index())
    fleet_daily.columns = ["date","n_anomalies","avg_health"]

    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=["Daily Anomaly Count","Fleet Avg Health Score"])
    fig2.add_trace(go.Bar(x=fleet_daily["date"], y=fleet_daily["n_anomalies"],
                           name="Anomalies", marker_color="#E24B4A", opacity=0.75),
                   row=1, col=1)
    fig2.add_trace(go.Scatter(x=fleet_daily["date"], y=fleet_daily["avg_health"],
                               line=dict(color="#1D9E75", width=2),
                               fill="tozeroy", fillcolor="rgba(29,158,117,0.1)",
                               name="Avg Health"),
                   row=2, col=1)
    fig2.add_hline(y=thresh_warning, row=2, line_dash="dash", line_color="#BA7517")
    fig2.update_layout(height=400, margin=dict(l=0,r=0,t=40,b=0),
                       showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Top 10 Machines by Anomaly Count")
    top_anom = (df_filt[df_filt["if_anomaly_01"]==1]
                .groupby("machineID").size()
                .sort_values(ascending=False)
                .head(10).reset_index())
    top_anom.columns = ["Machine ID","Anomaly Count"]
    st.dataframe(top_anom, use_container_width=True, height=300)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4: ANOMALY FORECAST
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Anomaly Forecast":
    st.title("🔮 Anomaly Forecast")
    st.caption(
        "Proyeksi potensi anomali, warnings, dan critical events ke depan. "
        "Menggunakan linear trend extrapolation (14-day lookback) + exponential smoothing anomaly rate."
    )

    # Filter forecast_df to machines in selected models
    fcast_view = forecast_df[
        forecast_df["machineID"].isin(df_filt["machineID"].unique())
    ].copy()

    # Apply threshold-aware risk re-classification
    def recompute_risk(row):
        score_min = min(row["forecast_72h"], row["forecast_7d"], row["current_score"])
        if score_min < thresh_critical or row["anom_rate_current"] > 0.5:
            return "Critical"
        elif score_min < thresh_warning or row["anom_rate_current"] > 0.35:
            return "High"
        elif score_min < thresh_warning + 10 or row["anom_rate_current"] > 0.20:
            return "Medium"
        return "Low"

    fcast_view["risk_level"] = fcast_view.apply(recompute_risk, axis=1)

    # ── Summary KPIs ────────────────────────────────────────────────────────
    st.subheader("📊 Forecast Summary")
    r_crit   = (fcast_view["risk_level"] == "Critical").sum()
    r_high   = (fcast_view["risk_level"] == "High").sum()
    r_med    = (fcast_view["risk_level"] == "Medium").sum()
    r_low    = (fcast_view["risk_level"] == "Low").sum()
    n_tot    = len(fcast_view)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔴 Critical Risk",  r_crit, delta=f"{r_crit/n_tot*100:.0f}% of fleet", delta_color="inverse")
    k2.metric("🟠 High Risk",      r_high, delta=f"{r_high/n_tot*100:.0f}% of fleet", delta_color="inverse")
    k3.metric("🟡 Medium Risk",    r_med,  delta=f"{r_med/n_tot*100:.0f}% of fleet",  delta_color="off")
    k4.metric("🟢 Low Risk",       r_low,  delta=f"{r_low/n_tot*100:.0f}% of fleet",  delta_color="off")

    st.divider()

    # ── Forecast Risk Table ──────────────────────────────────────────────────
    col_tbl, col_dist = st.columns([3, 2])

    with col_tbl:
        st.subheader("🗂️ Forecast Risk Table")
        st.caption("Sorted by forecast 7-day score (ascending = worst first)")

        tbl = fcast_view[[
            "machineID","current_score","forecast_72h","forecast_7d",
            "trend_slope","anom_rate_current","days_to_critical","risk_level"
        ]].copy()
        tbl = tbl.sort_values("forecast_7d")
        tbl["current_score"]      = tbl["current_score"].round(1)
        tbl["forecast_72h"]       = tbl["forecast_72h"].round(1)
        tbl["forecast_7d"]        = tbl["forecast_7d"].round(1)
        tbl["trend_slope"]        = tbl["trend_slope"].round(2)
        tbl["anom_rate_current"]  = (tbl["anom_rate_current"] * 100).round(1)
        tbl["days_to_critical"]   = tbl["days_to_critical"].apply(
            lambda x: f"{x:.1f}" if x < 365 else "—"
        )
        tbl["Risk"] = tbl["risk_level"].map(lambda r: f"{RISK_ICON.get(r,'❓')} {r}")
        tbl = tbl.drop(columns=["risk_level"])
        tbl = tbl.rename(columns={
            "machineID"         : "Machine",
            "current_score"     : "Now",
            "forecast_72h"      : "72h",
            "forecast_7d"       : "7d",
            "trend_slope"       : "Trend (pts/day)",
            "anom_rate_current" : "Anom Rate %",
            "days_to_critical"  : "Days→Critical",
        })
        st.dataframe(tbl, use_container_width=True, height=420)
        csv_fc = tbl.to_csv(index=False).encode("utf-8")
        st.download_button("Export Forecast CSV", csv_fc, "forecast_risk.csv", "text/csv")

    with col_dist:
        st.subheader("📊 Risk Distribution")

        # Donut risk
        risk_counts = fcast_view["risk_level"].value_counts().reindex(
            ["Critical","High","Medium","Low"], fill_value=0
        )
        fig_risk_pie = go.Figure(go.Pie(
            labels=risk_counts.index.tolist(),
            values=risk_counts.values.tolist(),
            marker_colors=["#E24B4A","#C0392B","#BA7517","#1D9E75"],
            hole=0.5,
            textinfo="percent+label",
        ))
        fig_risk_pie.update_layout(height=240, margin=dict(l=0,r=0,t=10,b=0),
                                    showlegend=False)
        st.plotly_chart(fig_risk_pie, use_container_width=True)

        # Now vs 7d scatter
        st.subheader("📉 Now vs Forecast 7d")
        fig_scatter = go.Figure()
        risk_order = ["Critical","High","Medium","Low"]
        for rl in risk_order:
            sub = fcast_view[fcast_view["risk_level"] == rl]
            if sub.empty: continue
            fig_scatter.add_trace(go.Scatter(
                x=sub["current_score"],
                y=sub["forecast_7d"],
                mode="markers",
                name=rl,
                marker=dict(color=RISK_COLOR[rl], size=8, opacity=0.8),
                text=sub["machineID"].astype(str),
                hovertemplate="Machine %{text}<br>Now: %{x:.1f}<br>7d: %{y:.1f}<extra>" + rl + "</extra>",
            ))
        # Diagonal reference line
        fig_scatter.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100],
            mode="lines",
            line=dict(color="gray", dash="dot", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_scatter.update_layout(
            xaxis_title="Current Score",
            yaxis_title="Forecast Score (7d)",
            height=240,
            margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ── Multi-machine projected health chart ──────────────────────────────────
    st.subheader("📈 Projected Health Score — Top At-Risk Machines")
    st.caption("Tren 14 hari terakhir + proyeksi 7 hari ke depan untuk 10 mesin dengan risiko tertinggi.")

    top_risk = fcast_view.sort_values("forecast_7d").head(10)["machineID"].tolist()

    with st.spinner("Generating forecast series..."):
        fig_multi = go.Figure()
        palette = px.colors.qualitative.Set2

        for i, mid in enumerate(top_risk):
            fc_s = compute_machine_forecast_series(mid, lookback_days=14)
            col_line = palette[i % len(palette)]
            # Historical (thinner, semi-transparent)
            fig_multi.add_trace(go.Scatter(
                x=fc_s["hist_dt"], y=fc_s["hist_score"],
                name=f"M{mid} hist", line=dict(color=col_line, width=1.2),
                opacity=0.4, showlegend=False,
                hoverinfo="skip",
            ))
            # Forecast
            fig_multi.add_trace(go.Scatter(
                x=fc_s["fut_dt_7d"], y=fc_s["fut_score_7d"],
                name=f"Machine {mid}",
                line=dict(color=col_line, width=2, dash="dot"),
                opacity=0.9,
                hovertemplate=f"Machine {mid}<br>%{{x}}<br>Score: %{{y:.1f}}<extra></extra>",
            ))

    fig_multi.add_hline(y=thresh_warning,  line_dash="dash", line_color="#1D9E75",
                        annotation_text="Warning")
    fig_multi.add_hline(y=thresh_critical, line_dash="dash", line_color="#E24B4A",
                        annotation_text="Critical")

    last_hist_dt_str = df["datetime"].max().isoformat()
    fig_multi.add_vline(x=last_hist_dt_str, line_dash="solid",
                         line_color="rgba(255,255,255,0.25)",
                         annotation_text="Now ↓ Forecast →")

    fig_multi.update_layout(
        height=400,
        yaxis_range=[0, 105],
        yaxis_title="Health Score",
        xaxis_title="Datetime",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_multi, use_container_width=True)

    st.divider()

    # ── 7-Day Warning Calendar ──────────────────────────────────────────────
    st.subheader("📅 7-Day Risk Calendar")
    st.caption("Heatmap prediksi status mesin per hari selama 7 hari ke depan. "
               "Warna merah = prediksi critical, kuning = warning, hijau = healthy.")

    last_dt  = df["datetime"].max()
    day_offsets = list(range(1, 8))
    day_labels  = [(last_dt + pd.Timedelta(days=d)).strftime("Day+%d\n%b %d") for d in day_offsets]

    # Build calendar matrix: machines (rows) × days (cols)
    cal_machines = fcast_view.sort_values("forecast_7d").head(20)["machineID"].tolist()

    calendar_z    = []   # numeric: 0=Healthy,1=Warning,2=Critical
    calendar_text = []   # hover text

    for mid in cal_machines:
        fc_s = compute_machine_forecast_series(mid, lookback_days=14)
        fut_scores = fc_s["fut_score_7d"]  # 168 hourly values
        row_z    = []
        row_text = []
        for d in day_offsets:
            # Average score for that day (24-hour window)
            idx_start = (d-1) * 24
            idx_end   = d * 24
            day_vals  = fut_scores[idx_start:min(idx_end, len(fut_scores))]
            avg_score = float(np.mean(day_vals)) if day_vals else 50.0
            if avg_score < thresh_critical:
                z_val = 2
            elif avg_score < thresh_warning:
                z_val = 1
            else:
                z_val = 0
            row_z.append(z_val)
            row_text.append(f"Machine {mid}<br>Day+{d}: {avg_score:.1f}")
        calendar_z.append(row_z)
        calendar_text.append(row_text)

    fig_cal = go.Figure(go.Heatmap(
        z=calendar_z,
        x=day_labels,
        y=[f"M{mid}" for mid in cal_machines],
        text=calendar_text,
        hovertemplate="%{text}<extra></extra>",
        colorscale=[
            [0.0,  "#1D9E75"],  # Healthy
            [0.5,  "#BA7517"],  # Warning
            [1.0,  "#E24B4A"],  # Critical
        ],
        showscale=True,
        colorbar=dict(
            title="Risk",
            tickvals=[0, 1, 2],
            ticktext=["Healthy", "Warning", "Critical"],
            len=0.8,
        ),
        zmin=0, zmax=2,
        xgap=3, ygap=3,
    ))
    fig_cal.update_layout(
        height=500,
        margin=dict(l=0, r=80, t=10, b=0),
        xaxis=dict(side="top"),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    st.divider()

    # ── Anomaly Rate Trend ──────────────────────────────────────────────────
    st.subheader("📡 Current vs Projected Anomaly Rate")
    st.caption("Rata-rata anomaly rate saat ini vs proyeksi 7 hari ke depan per mesin (top 20 at-risk).")

    anom_sub = fcast_view.sort_values("forecast_7d").head(20).copy()
    anom_sub["anom_rate_pct"]   = (anom_sub["anom_rate_current"] * 100).round(1)
    anom_sub["anom_rate_7d_pct"]= (anom_sub["anom_rate_7d"] * 100).round(1)

    fig_anom = go.Figure()
    fig_anom.add_trace(go.Bar(
        x=anom_sub["machineID"].astype(str),
        y=anom_sub["anom_rate_pct"],
        name="Current Anom Rate %",
        marker_color="#185FA5",
        opacity=0.8,
    ))
    fig_anom.add_trace(go.Bar(
        x=anom_sub["machineID"].astype(str),
        y=anom_sub["anom_rate_7d_pct"],
        name="Projected Anom Rate % (7d)",
        marker_color="#E24B4A",
        opacity=0.7,
    ))
    fig_anom.update_layout(
        barmode="group",
        xaxis_title="Machine ID",
        yaxis_title="Anomaly Rate (%)",
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_anom, use_container_width=True)
