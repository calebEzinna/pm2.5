"""
Loads:
    best_rf_model.pkl   — RandomForestClassifier  (alert classification)
    best_reg_model.pkl  — LinearRegression or RF wrapped as dict {model, scaler, type}
    scaler.pkl          — StandardScaler (classification pipeline)
    
"""

import io, re, warnings
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# Constants 
ALERT_THRESHOLD = 15

MONTH_OPTIONS = ["January","February","March","April","May","June",
                 "July","August","September","October","November","December"]
MONTH_TO_NUM  = {m: i+1 for i, m in enumerate(MONTH_OPTIONS)}
DAY_OPTIONS   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
HOUR_OPTIONS  = [f"{h:02d}:00" for h in range(24)]

AQI_BANDS = [
    (0,   5,   "Excellent (WHO)",  "#1abc9c"),
    (5,   15,  "Good",             "#27ae60"),
    (15,  25,  "Moderate",         "#f1c40f"),
    (25,  55,  "Unhealthy (S.G.)","#e67e22"),
    (55,  150, "Unhealthy",        "#e74c3c"),
    (150, 999, "Hazardous",        "#8e44ad"),
]

BASE_FEATURES = [
    "station_mean_pm25","station_std_pm25","missing_pm25",
    "hour","month","hour_sin","hour_cos",
    "lag_1","lag_3","lag_6","lag_12",
    "roll_3","roll_6","roll_12","roll_24",
]

ALL_STATIONS = [
    "Aberdeen Anderson Dr","Aberdeen Erroll Place","Aberdeen Erroll Park",
    "Aberdeen King Street","Aberdeen Market Street 2",
    "Aberdeen Union Street Roadside","Aberdeen Wellington Road",
    "Auchencorth Moss",
    "Dundee Broughty Ferry Road","Dundee Lochee Road","Dundee Mains Loan",
    "Dundee Meadowside","Dundee Seagate","Dundee Whitehall Street",
    "Edinburgh Currie","Edinburgh Drumsheugh","Edinburgh Glasgow Road",
    "Edinburgh Nicolson Street","Edinburgh Queensferry Road",
    "Edinburgh Salamander St","Edinburgh St John's Road",
    "Edinburgh St Leonards","Edinburgh Tower Street",
    "Glasgow Anderston","Glasgow Broomhill","Glasgow Burgher St",
    "Glasgow Byres Road","Glasgow Dumbarton Road","Glasgow High Street",
    "Glasgow Kerbside","Glasgow Nithsdale Road","Glasgow Townhead",
    "Glasgow Waulkmilglen Reservoir",
    "West Lothian Broxburn","West Lothian Linlithgow High Street 2",
    "West Lothian Newton",
]
STATION_DUMMIES = ["station_"+re.sub(r"[^A-Za-z0-9_]+","_",s) for s in ALL_STATIONS[1:]]
ALL_FEATURE_COLS = BASE_FEATURES + STATION_DUMMIES


def aqi_label(val):
    for lo, hi, label, colour in AQI_BANDS:
        if lo <= val < hi:
            return label, colour
    return "Hazardous", "#8e44ad"


# Page config & styling
st.set_page_config(page_title="PM2.5 Dashboard", page_icon="🌫️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .stApp{background-color:#0f1117;color:#e0e0e0}
  [data-testid="metric-container"]{background:#1e2130;border:1px solid #2d3250;
      border-radius:10px;padding:12px 16px}
  [data-testid="stSidebar"]{background-color:#161b27}
  .alert-box{background:#3d1515;border-left:5px solid #e74c3c;
      border-radius:6px;padding:14px 18px;margin:8px 0}
  .safe-box{background:#0d2b1a;border-left:5px solid #2ecc71;
      border-radius:6px;padding:14px 18px;margin:8px 0}
  .warn-box{background:#2b2200;border-left:5px solid #f1c40f;
      border-radius:6px;padding:14px 18px;margin:8px 0}
  .info-box{background:#0d1f33;border-left:5px solid #3498db;
      border-radius:6px;padding:12px 16px;margin:8px 0;font-size:0.92rem}
  .section-title{font-size:1.1rem;font-weight:700;color:#7eb8f7;
      margin-bottom:4px;margin-top:8px}
</style>
""", unsafe_allow_html=True)


#  Model loaders

@st.cache_resource
def load_all_models():
    """Load all models directly . Cached ."""
    # Classifier
    clf    = joblib.load("the_best_rf_model.pkl")
    scaler = joblib.load("the_scaler.pkl")
    feat   = list(clf.feature_names_in_) if hasattr(clf, "feature_names_in_") else None

    # Regressor
    obj = joblib.load("best_reg_model.pkl")
    if isinstance(obj, dict):
        reg      = obj["model"]
        reg_sc   = obj.get("scaler")
        reg_feat = None
        reg_type = obj.get("type", "linear")
    else:
        reg      = obj
        reg_sc   = None
        reg_feat = list(obj.feature_names_in_) if hasattr(obj, "feature_names_in_") else None
        reg_type = "rf"

    return clf, scaler, feat, reg, reg_feat, reg_sc, reg_type


# Sidebar

with st.sidebar:
    st.markdown("## 🌫️ PM2.5 Dashboard")
    st.markdown("*Direct model inference*")
    st.markdown("---")
    st.markdown("### Models")
    st.caption("Loaded automatically from repository")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🎛️ Manual Predict", "📂 Batch Predict",
        "🔬 Model Inspector", "📖 Feature Guide",
    ], label_visibility="collapsed")
    st.markdown("---")
    threshold = st.slider("Alert threshold (µg/m³)", 5, 50, ALERT_THRESHOLD)
    st.caption("WHO 2021 24-hour guideline = 15 µg/m³")


# Load models

with st.spinner("Loading models…"):
    clf, clf_scaler, clf_feat, reg, reg_feat, reg_scaler, reg_type = load_all_models()

st.sidebar.success(f"✅ Classifier: {type(clf).__name__}")
st.sidebar.caption(f"{len(clf_feat) if clf_feat else '?'} features")
if reg is not None:
    _rt = reg.get("type","rf") if isinstance(reg,dict) else (reg_type or "rf")
    st.sidebar.success(f"✅ Regressor: {type(reg).__name__ if not isinstance(reg,dict) else type(reg['model']).__name__} ({_rt})")


def require_clf():
    pass  


# Core helpers

def build_feature_row(values: dict, station: str, feat_cols=None) -> pd.DataFrame:
    """Build one feature row, computing hour_sin/cos automatically."""
    hour  = int(values.get("hour", 12))
    month = int(values.get("month", 6))
    values["hour_sin"] = float(np.sin(2 * np.pi * hour  / 24))
    values["hour_cos"] = float(np.cos(2 * np.pi * hour  / 24))
    row = {k: float(values.get(k, 0.0)) for k in BASE_FEATURES}
    if station:
        row["station_" + re.sub(r"[^A-Za-z0-9_]+", "_", station)] = 1.0
    df_row = pd.DataFrame([row])
    cols   = feat_cols if feat_cols else ALL_FEATURE_COLS
    for col in cols:
        if col not in df_row.columns:
            df_row[col] = 0.0
    return df_row[cols].fillna(0.0)


def predict_alert(df_row: pd.DataFrame) -> dict:
    """Return alert_prob and alert_pred from the classifier."""
    prob = float(clf.predict_proba(df_row)[0, 1])
    pred = int(clf.predict(df_row)[0])
    return {"alert_prob": prob, "alert_pred": pred}


def predict_pm25(df_row: pd.DataFrame, fallback_roll24: float) -> float:
    """
    Return predicted PM2.5 concentration.

    """
    if reg is None:
        return float(np.clip(fallback_roll24, 0.0, 999.0))

    try:
        # Unwrap dict format defensively
        if isinstance(reg, dict):
            _model  = reg["model"]
            _scaler = reg.get("scaler")
            _type   = reg.get("type", "linear")
        else:
            _model  = reg
            _scaler = reg_scaler
            _type   = reg_type if reg_type else "rf"

        # Align feature columns
        feat_cols  = reg_feat if reg_feat else (clf_feat or list(df_row.columns))
        df_aligned = df_row.copy()
        for col in feat_cols:
            if col not in df_aligned.columns:
                df_aligned[col] = 0.0
        X_in = df_aligned[feat_cols].fillna(0.0)

        # Apply scaler for linear regression
        if _type == "linear" and _scaler is not None:
            X_in = _scaler.transform(X_in)

        return float(np.clip(float(_model.predict(X_in)[0]), 0.0, 999.0))

    except Exception:
        return float(np.clip(fallback_roll24, 0.0, 999.0))


def advance_hour(hour: int, month: int):
    try:
        dt = datetime(2026, month, 1, hour) + timedelta(hours=1)
        return dt.hour, dt.month
    except Exception:
        return (hour + 1) % 24, month


def compute_lag_roll(history: list) -> dict:
    def lag(k):  return history[-k] if len(history) >= k else history[0]
    def roll(w): return float(np.mean(history[-w:] if len(history) >= w else history))
    return {
        "lag_1": lag(1), "lag_3": lag(3), "lag_6": lag(6), "lag_12": lag(12),
        "roll_3": roll(3), "roll_6": roll(6), "roll_12": roll(12), "roll_24": roll(24),
    }


def reg_source_label():
    _type = reg.get("type","rf") if isinstance(reg, dict) else (reg_type or "rf")
    return "Linear Regression" if _type == "linear" else "RF Regressor"


# PAGE 1 — MANUAL PREDICT

if page == "🎛️ Manual Predict":

    require_clf()

    mode = st.radio("Prediction mode",
                    ["🔍 Single Observation", "📈 Future PM2.5 Forecast"],
                    horizontal=True)
    st.markdown("---")

    # Shared time/location selectors
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    station    = c1.selectbox("Monitoring Station", ALL_STATIONS, index=14)
    day_name   = c2.selectbox("Day of week", DAY_OPTIONS, index=0,
                               help="Shown in output for context — not a model feature.")
    month_name = c3.selectbox("Month", MONTH_OPTIONS, index=6)
    hour_label = c4.selectbox("Hour of day", HOUR_OPTIONS, index=14)
    month_num  = MONTH_TO_NUM[month_name]
    hour_num   = int(hour_label.split(":")[0])

    h_sin = float(np.sin(2 * np.pi * hour_num / 24))
    h_cos = float(np.cos(2 * np.pi * hour_num / 24))
    st.caption(
        f"**{day_name}, {hour_label}, {month_name}** at **{station}**  ·  "
        f"hour_sin = {h_sin:.4f}, hour_cos = {h_cos:.4f}"
    )



    st.markdown("---")

    # Station statistics
    st.markdown('<div class="section-title">📍 Station Background</div>',
                unsafe_allow_html=True)
    st.caption("Long-run historical stats — from the notebook station summary output.")
    c1, c2, c3 = st.columns(3)
    station_mean = c1.number_input("Station mean PM2.5 (µg/m³)", 0.0, 200.0, 8.0, 0.1, key="sm")
    station_std  = c2.number_input("Station std PM2.5 (µg/m³)",  0.0, 100.0, 4.0, 0.1, key="ss")
    missing_flag = c3.selectbox("Last reading missing?", [0, 1],
                                 format_func=lambda x: "No — sensor active" if x==0 else "Yes — sensor was down",
                                 key="mf")

    st.markdown("---")

    # Recent PM2.5 history
    st.markdown('<div class="section-title">⏮️ Recent PM2.5 History</div>',
                unsafe_allow_html=True)
    st.caption("Readings from the hours before this observation — used for lag and rolling mean features.")
    c1, c2, c3, c4 = st.columns(4)
    h1  = c1.number_input("1 hour ago (µg/m³)",   0.0, 999.0, 7.5, 0.1, key="h1")
    h3  = c2.number_input("3 hours ago (µg/m³)",  0.0, 999.0, 7.2, 0.1, key="h3")
    h6  = c3.number_input("6 hours ago (µg/m³)",  0.0, 999.0, 6.9, 0.1, key="h6")
    h12 = c4.number_input("12 hours ago (µg/m³)", 0.0, 999.0, 6.5, 0.1, key="h12")

    # Auto-compute rolling means from history
    history_seed = [h12]*6 + [h6]*6 + [h3]*3 + [h1]*3 + [h1]*6
    r3  = float(np.mean(history_seed[-3:]))
    r6  = float(np.mean(history_seed[-6:]))
    r12 = float(np.mean(history_seed[-12:]))
    r24 = float(np.mean(history_seed[-24:]))

    with st.expander("📊 Rolling means (auto-computed — expand to override)"):
        st.caption("Derived from your history inputs. Override only if you have exact values.")
        c1, c2, c3, c4 = st.columns(4)
        r3  = c1.number_input("3-hour mean",  0.0, 999.0, round(r3,  2), 0.1, key="r3")
        r6  = c2.number_input("6-hour mean",  0.0, 999.0, round(r6,  2), 0.1, key="r6")
        r12 = c3.number_input("12-hour mean", 0.0, 999.0, round(r12, 2), 0.1, key="r12")
        r24 = c4.number_input("24-hour mean", 0.0, 999.0, round(r24, 2), 0.1, key="r24")

    base_values = {
        "hour": hour_num, "month": month_num,
        "station_mean_pm25": station_mean, "station_std_pm25": station_std,
        "missing_pm25": missing_flag,
        "lag_1": h1, "lag_3": h3, "lag_6": h6, "lag_12": h12,
        "roll_3": r3, "roll_6": r6, "roll_12": r12, "roll_24": r24,
    }

    st.markdown("---")

    # SINGLE OBSERVATION
    if mode == "🔍 Single Observation":

        if st.button("🔮 Predict Now", type="primary", use_container_width=True):
            df_row = build_feature_row(base_values, station, clf_feat)
            alert  = predict_alert(df_row)
            pm25   = predict_pm25(df_row, r24)

            alert_prob = alert["alert_prob"]
            alert_pred = alert["alert_pred"]
            aqi_lbl, aqi_clr = aqi_label(pm25)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted PM2.5",   f"{pm25:.1f} µg/m³",
                      help=f"Source: {reg_source_label()}")
            m2.metric("Alert Probability", f"{alert_prob*100:.1f}%")
            m3.metric("Alert Prediction",  "🚨 ALERT" if alert_pred else "✅ SAFE")
            m4.metric("AQI Band",          aqi_lbl)

            if alert_pred == 1 or pm25 > threshold:
                st.markdown(
                    f'<div class="alert-box">🚨 <b>Pollution alert predicted.</b> '
                    f'PM2.5: <b>{pm25:.1f} µg/m³</b> — exceeds {threshold} µg/m³. '
                    f'Alert probability: <b>{alert_prob*100:.1f}%</b>.</div>',
                    unsafe_allow_html=True)
            elif alert_prob > 0.3:
                st.markdown(
                    f'<div class="warn-box">⚠️ <b>Elevated risk.</b> '
                    f'PM2.5: <b>{pm25:.1f} µg/m³</b>. '
                    f'Alert probability: <b>{alert_prob*100:.1f}%</b>.</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="safe-box">✅ <b>No alert predicted.</b> '
                    f'PM2.5: <b>{pm25:.1f} µg/m³</b>. '
                    f'Alert probability: <b>{alert_prob*100:.1f}%</b>.</div>',
                    unsafe_allow_html=True)



            col_g, col_p = st.columns(2)
            with col_g:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=alert_prob*100,
                    number={"suffix":"%","font":{"size":34}},
                    title={"text":"Alert Probability","font":{"size":15}},
                    gauge={
                        "axis":{"range":[0,60],"tickcolor":"#aaa"},
                        "bar":{"color":"#e74c3c" if alert_prob>0.5 else "#f1c40f" if alert_prob>0.3 else "#2ecc71"},
                        "steps":[{"range":[0,5],  "color":"#0d3b2b"},
                                 {"range":[5,15],  "color":"#0d2b1a"},
                                 {"range":[15,25], "color":"#2b2200"},
                                 {"range":[25,60], "color":"#3d1515"}],
                        "threshold":{"line":{"color":"white","width":3},"thickness":0.8,"value":50},
                    },
                ))
                fig_g.update_layout(template="plotly_dark", height=280,
                                    margin=dict(t=40,b=10,l=20,r=20))
                st.plotly_chart(fig_g, use_container_width=True)

            with col_p:
                fig_p = go.Figure(go.Indicator(
                    mode="gauge+number", value=pm25,
                    number={"suffix":" µg/m³","font":{"size":26}},
                    title={"text":f"Predicted PM2.5 — {aqi_lbl}","font":{"size":14}},
                    gauge={
                        "axis":{"range":[0,60],"tickcolor":"#aaa"},
                        "bar":{"color":aqi_clr},
                        "steps":[{"range":[0,5],  "color":"#0d3b2b"},
                                 {"range":[5,15],  "color":"#0d2b1a"},
                                 {"range":[15,25], "color":"#2b2200"},
                                 {"range":[25,60], "color":"#3d1515"}],
                        "threshold":{"line":{"color":"white","width":3},"thickness":0.8,"value":threshold},
                    },
                ))
                fig_p.update_layout(template="plotly_dark", height=280,
                                    margin=dict(t=40,b=10,l=20,r=20))
                st.plotly_chart(fig_p, use_container_width=True)

            with st.expander("🔍 Feature vector sent to model"):
                st.dataframe(df_row.T.rename(columns={0:"value"}), use_container_width=True)

    # FUTURE FORECAST
    else:
        st.markdown("### 📈 Future PM2.5 Forecast")
        st.markdown("Rolls forward hour by hour — each prediction feeds back as a lag for the next step.")

        current_pm25 = st.number_input("Current PM2.5 right now (µg/m³)", 0.0, 999.0, h1, 0.1, key="cur")
        horizon_lbl  = st.selectbox("Forecast horizon",
                                     ["6 hours","12 hours","24 hours","48 hours","72 hours"],
                                     index=2)
        horizon = int(horizon_lbl.split()[0])
        st.markdown("---")

        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):

            seed = [h12]*6 + [h6]*6 + [h3]*3 + [h1]*2 + [current_pm25]
            rolling = list(seed)
            cur_hr, cur_mo = hour_num, month_num
            # Track day of week — advances every time the hour rolls past midnight
            day_idx = DAY_OPTIONS.index(day_name)   # 0=Mon … 6=Sun
            rows = []

            with st.spinner(f"Generating {horizon}-hour forecast…"):
                for step in range(1, horizon+1):
                    prev_hr = cur_hr
                    cur_hr, cur_mo = advance_hour(cur_hr, cur_mo)
                    # Midnight crossed — advance day of week
                    if cur_hr < prev_hr:
                        day_idx = (day_idx + 1) % 7
                    cur_day_name = DAY_OPTIONS[day_idx]

                    lr = compute_lag_roll(rolling)
                    fv = {"hour": cur_hr, "month": cur_mo,
                          "station_mean_pm25": station_mean,
                          "station_std_pm25":  station_std,
                          "missing_pm25": 0, **lr}
                    df_row    = build_feature_row(fv, station, clf_feat)
                    alert     = predict_alert(df_row)
                    pm25_fc   = predict_pm25(df_row, lr["roll_24"])
                    aqi_l, aqi_c = aqi_label(pm25_fc)
                    rows.append({
                        "Step": step, "Day": cur_day_name,
                        "Month": MONTH_OPTIONS[cur_mo-1],
                        "Hour": f"{cur_hr:02d}:00",
                        "PM2.5 (µg/m³)": round(pm25_fc, 2),
                        "Alert Prob (%)": round(alert["alert_prob"]*100, 2),
                        "Alert": "🚨 YES" if (alert["alert_pred"] or pm25_fc>threshold) else "✅ NO",
                        "AQI Band": aqi_l,
                        "_c": aqi_c,
                    })
                    rolling.append(pm25_fc)

            fdf = pd.DataFrame(rows)
            st.caption(f"PM2.5 source: **{reg_source_label()}**")

            # KPIs
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Peak PM2.5",    f"{fdf['PM2.5 (µg/m³)'].max():.1f} µg/m³")
            c2.metric("Mean PM2.5",    f"{fdf['PM2.5 (µg/m³)'].mean():.1f} µg/m³")
            c3.metric("Alert hours",    f"{(fdf['Alert']=='🚨 YES').sum()} / {horizon}")
            peak_row = fdf.loc[fdf["Alert Prob (%)"].idxmax()]
            c4.metric("Peak alert prob",f"{peak_row['Alert Prob (%)']:.1f}% at {peak_row['Hour']}")

            # PM2.5 chart
            x = [f"H+{r['Step']} ({r['Hour']})" for _, r in fdf.iterrows()]
            fig_fc = go.Figure()
            fig_fc.add_hrect(y0=threshold, y1=max(fdf["PM2.5 (µg/m³)"].max()*1.15, threshold+5),
                             fillcolor="rgba(231,76,60,0.07)", line_width=0,
                             annotation_text="Alert zone", annotation_position="top left")
            fig_fc.add_trace(go.Scatter(
                x=x, y=fdf["PM2.5 (µg/m³)"], name="Predicted PM2.5",
                mode="lines+markers", line=dict(color="#7eb8f7", width=2.5),
                marker=dict(size=8, color=fdf["_c"].tolist(), line=dict(color="white",width=1)),
                fill="tozeroy", fillcolor="rgba(126,184,247,0.07)",
            ))
            fig_fc.add_hline(y=threshold, line_dash="dash", line_color="#e74c3c",
                             annotation_text=f"WHO threshold ({threshold} µg/m³)",
                             annotation_position="top right")
            fig_fc.add_hline(y=threshold, line_dash="dash", line_color="#e74c3c",
                 annotation_text=f"Alert threshold ({threshold} µg/m³)",
                 annotation_position="top right")
            fig_fc.update_layout(
                template="plotly_dark", height=400,
                title=f"{horizon}-Hour Forecast — {station} — {day_name} {hour_label}, {month_name}",
                xaxis_title="Forecast step", yaxis_title="PM2.5 (µg/m³)",
                xaxis=dict(tickangle=-40), margin=dict(t=50,b=10),
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Alert probability bar
            fig_p = go.Figure(go.Bar(
                x=x, y=fdf["Alert Prob (%)"],
                marker_color=["#e74c3c" if p>=50 else "#f1c40f" if p>=30 else "#2ecc71"
                              for p in fdf["Alert Prob (%)"]],
            ))
            fig_p.add_hline(y=50, line_dash="dash", line_color="white",
                            annotation_text="50% threshold")
            fig_p.update_layout(template="plotly_dark", height=250,
                                title="Alert Probability per Hour",
                                xaxis_title="", yaxis_title="Alert Probability (%)",
                                xaxis=dict(tickangle=-40), margin=dict(t=40,b=10))
            st.plotly_chart(fig_p, use_container_width=True)

            # Table + download
            dcols = ["Step","Day","Month","Hour","PM2.5 (µg/m³)","Alert Prob (%)","Alert","AQI Band"]
            st.dataframe(
                fdf[dcols].style.map(
                    lambda v: "background-color:#3d1515;color:#ff9999" if v=="🚨 YES"
                    else "background-color:#0d2b1a;color:#99ffbb" if v=="✅ NO" else "",
                    subset=["Alert"]),
                use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download CSV",
                               fdf[dcols].to_csv(index=False).encode(),
                               f"pm25_forecast_{station.replace(' ','_')}_{horizon}h.csv",
                               "text/csv")

            n_alerts = (fdf["Alert"]=="🚨 YES").sum()
            if n_alerts > 0:
                st.markdown(
                    f'<div class="alert-box">🚨 <b>{n_alerts} alert hour(s) in next {horizon} hours.</b> '
                    f'Peak PM2.5: <b>{fdf["PM2.5 (µg/m³)"].max():.1f} µg/m³</b>.</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="safe-box">✅ <b>No alerts in next {horizon} hours.</b> '
                    f'Peak: <b>{fdf["PM2.5 (µg/m³)"].max():.1f} µg/m³</b> — below {threshold} µg/m³.</div>',
                    unsafe_allow_html=True)


# PAGE 2 — BATCH PREDICT

elif page == "📂 Batch Predict":
    st.markdown("# 📂 Batch Prediction")
    require_clf()

    if clf_feat:
        t = pd.DataFrame(columns=clf_feat)
        t.loc[0] = 0.0
        for col,val in [("station_mean_pm25",8.0),("station_std_pm25",4.0),
                         ("missing_pm25",0),("hour",14),("month",7),
                         ("hour_sin",float(np.sin(2*np.pi*14/24))),
                         ("hour_cos",float(np.cos(2*np.pi*14/24))),
                         ("lag_1",7.5),("lag_3",7.2),("lag_6",6.9),("lag_12",6.5),
                         ("roll_3",7.4),("roll_6",7.2),("roll_12",7.0),("roll_24",6.8)]:
            if col in t.columns: t.loc[0,col] = val
        st.download_button("⬇️ Download feature template",
                           t.to_csv(index=False).encode(),
                           "pm25_feature_template.csv","text/csv")

    batch_file = st.file_uploader("Upload feature CSV", type="csv")
    if batch_file:
        try:
            bdf = pd.read_csv(batch_file)
            st.success(f"Loaded {len(bdf):,} rows × {bdf.shape[1]} columns.")
            if clf_feat:
                for col in [c for c in clf_feat if c not in bdf.columns]:
                    bdf[col] = 0.0
                clf_in = bdf[clf_feat].fillna(0.0)
            else:
                clf_in = bdf.fillna(0.0)

            if st.button("🔮 Predict All", type="primary"):
                probs = clf.predict_proba(clf_in)[:,1]
                preds = clf.predict(clf_in)

                pm25_vals = None
                if reg is not None:
                    try:
                        # Unwrap dict defensively
                        _m = reg["model"] if isinstance(reg,dict) else reg
                        _s = reg.get("scaler") if isinstance(reg,dict) else reg_scaler
                        _t = reg.get("type","rf") if isinstance(reg,dict) else (reg_type or "rf")
                        fc = reg_feat or clf_feat or list(bdf.columns)
                        ri = bdf.copy()
                        for col in fc:
                            if col not in ri.columns: ri[col] = 0.0
                        X_r = ri[fc].fillna(0.0)
                        if _t == "linear" and _s is not None:
                            X_r = _s.transform(X_r)
                        pm25_vals = np.clip(_m.predict(X_r), 0, 999)
                    except Exception:
                        pass
                elif "roll_24" in bdf.columns:
                    pm25_vals = bdf["roll_24"].values

                out = bdf.copy()
                out["alert_probability_%"] = np.round(probs*100, 2)
                out["alert_prediction"]    = preds
                out["alert_label"]         = np.where(preds==1,"ALERT","SAFE")
                if pm25_vals is not None:
                    out["predicted_pm25"] = np.round(pm25_vals, 2)
                    out["aqi_band"]       = [aqi_label(v)[0] for v in pm25_vals]

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Total rows",       f"{len(out):,}")
                c2.metric("Alerts predicted", f"{preds.sum():,}")
                c3.metric("Alert rate",       f"{preds.mean()*100:.2f}%")
                c4.metric("Mean alert prob",  f"{probs.mean()*100:.2f}%")

                fig_d = go.Figure(go.Histogram(x=probs*100, nbinsx=50,
                                               marker_color="#7eb8f7", opacity=0.8))
                fig_d.add_vline(x=50, line_dash="dash", line_color="#e74c3c")
                fig_d.update_layout(template="plotly_dark", height=260,
                                    title="Distribution of Alert Probabilities",
                                    xaxis_title="Alert Probability (%)",
                                    margin=dict(t=40,b=10))
                st.plotly_chart(fig_d, use_container_width=True)

                dcols = ["alert_probability_%","alert_prediction","alert_label"]
                if "predicted_pm25" in out.columns:
                    dcols = ["predicted_pm25","aqi_band"] + dcols
                if "roll_24" in out.columns:
                    dcols = ["roll_24"] + dcols
                st.dataframe(out[dcols].head(200), use_container_width=True)
                st.download_button("⬇️ Download results",
                                   out.to_csv(index=False).encode(),
                                   "pm25_batch_predictions.csv","text/csv")
        except Exception as e:
            st.error(f"Error: {e}")


# PAGE 3 — MODEL INSPECTOR

elif page == "🔬 Model Inspector":
    st.markdown("# 🔬 Model Inspector")
    require_clf()

    def inspector_panel(mdl, feat, label):
        st.markdown(f"### {label}")
        params = mdl.get_params() if hasattr(mdl,"get_params") else {}
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Type",         type(mdl).__name__)
        c2.metric("Features",     len(feat) if feat else "N/A")
        c3.metric("n_estimators", params.get("n_estimators","N/A"))
        c4.metric("max_depth",    params.get("max_depth","N/A"))
        with st.expander("All parameters"):
            st.dataframe(pd.DataFrame(list(params.items()),columns=["Parameter","Value"]),
                         use_container_width=True, hide_index=True)

        if hasattr(mdl,"feature_importances_") and feat:
            imp = (pd.DataFrame({"Feature":feat,"Importance":mdl.feature_importances_})
                   .sort_values("Importance",ascending=False).reset_index(drop=True))
            imp["Rank"] = imp.index+1
            top_n = st.slider("Top N features", 5, min(len(imp),50), 20,
                               key=f"topn_{label}")
            top = imp.head(top_n)
            ca, ct = st.columns([3,2])
            with ca:
                fig = px.bar(top, x="Importance", y="Feature", orientation="h",
                             template="plotly_dark",
                             color="Importance",
                             color_continuous_scale=["#3498db","#e74c3c"],
                             height=max(350,top_n*22))
                fig.update_layout(yaxis=dict(autorange="reversed"),
                                  coloraxis_showscale=False, margin=dict(t=10,b=10))
                st.plotly_chart(fig, use_container_width=True)
            with ct:
                st.dataframe(top[["Rank","Feature","Importance"]].style.format({"Importance":"{:.6f}"}),
                             use_container_width=True, hide_index=True,
                             height=max(350,top_n*22))

            # Group pie
            def fg(n):
                if n.startswith("lag_"):    return "Lag Features"
                if n.startswith("roll_"):   return "Rolling Means"
                if n.startswith("hour"):    return "Temporal (Hour)"
                if n.startswith("month"):   return "Temporal (Month)"
                if n in ("station_mean_pm25","station_std_pm25"): return "Station Stats"
                if n.startswith("station_"): return "Station Dummies"
                return "Other"
            imp["Group"] = imp["Feature"].apply(fg)
            grp = imp.groupby("Group")["Importance"].sum().reset_index().sort_values("Importance",ascending=False)
            fig_g = px.pie(grp, names="Group", values="Importance",
                           template="plotly_dark",
                           color_discrete_sequence=px.colors.qualitative.Set2, height=320)
            fig_g.update_traces(textposition="inside", textinfo="percent+label")
            fig_g.update_layout(margin=dict(t=10,b=10), showlegend=False)
            st.plotly_chart(fig_g, use_container_width=True)

    tab_clf, tab_reg = st.tabs(["🎯 Classifier", "📐 Regressor"])
    with tab_clf:
        inspector_panel(clf, clf_feat, "Classifier")
        st.markdown("---")
        st.markdown('<div class="section-title">Sensitivity Analysis</div>',
                    unsafe_allow_html=True)
        sweepable  = [f for f in BASE_FEATURES if f not in ("hour_sin","hour_cos")]
        sw_feat    = st.selectbox("Feature to sweep", sweepable,
                                   index=sweepable.index("roll_24") if "roll_24" in sweepable else 0)
        sw_station = st.selectbox("Station", ALL_STATIONS, index=0, key="sw_st")
        meta_lo    = {"lag_1":0,"lag_3":0,"lag_6":0,"lag_12":0,
                      "roll_3":0,"roll_6":0,"roll_12":0,"roll_24":0}.get(sw_feat, 0)
        meta_hi    = {"lag_1":120,"lag_3":120,"lag_6":120,"lag_12":120,
                      "roll_3":100,"roll_6":100,"roll_12":100,"roll_24":100,
                      "station_mean_pm25":200,"station_std_pm25":100,
                      "hour":23,"month":12}.get(sw_feat, 100)
        sweep_vals = np.linspace(meta_lo, meta_hi, 80)
        base = {"station_mean_pm25":8.0,"station_std_pm25":4.0,"missing_pm25":0,
                "hour":14,"month":7,"lag_1":7.5,"lag_3":7.2,"lag_6":6.9,"lag_12":6.5,
                "roll_3":7.4,"roll_6":7.2,"roll_12":7.0,"roll_24":6.8}
        sprobs = []
        for v in sweep_vals:
            b = base.copy(); b[sw_feat] = float(v)
            row = build_feature_row(b, sw_station, clf_feat)
            sprobs.append(float(clf.predict_proba(row)[0,1]))
        fig_s = go.Figure(go.Scatter(x=sweep_vals, y=sprobs, mode="lines",
                                     line=dict(color="#7eb8f7",width=2.5)))
        fig_s.add_hline(y=0.5, line_dash="dash", line_color="#e74c3c",
                        annotation_text="50% decision boundary")
        fig_s.update_layout(template="plotly_dark", height=280,
                             xaxis_title=sw_feat, yaxis_title="Alert Probability",
                             margin=dict(t=10,b=10))
        st.plotly_chart(fig_s, use_container_width=True)

    with tab_reg:
        _reg_mdl = reg["model"] if isinstance(reg,dict) else reg
        _reg_feat = reg_feat
        if _reg_mdl is not None:
            inspector_panel(_reg_mdl, _reg_feat, "Regressor")
        else:
            st.info("No regression model loaded. Upload best_reg_model.pkl in the sidebar.")


# PAGE 4 — FEATURE GUIDE

elif page == "📖 Feature Guide":
    st.markdown("# 📖 Feature Guide")

    st.markdown('<div class="section-title">Base Features</div>', unsafe_allow_html=True)
    guide = [{"Feature":f,"Range":f"{m[1]}–{m[2]}","Default":m[3],"Description":m[4]}
             for f,m in {
                 "station_mean_pm25":("",0,200,8,"Station historical mean PM2.5 µg/m³"),
                 "station_std_pm25": ("",0,100,4,"Station historical std dev µg/m³"),
                 "missing_pm25":     ("",0,1,0,"1 if sensor was down, 0 if normal"),
                 "hour":             ("",0,23,12,"Hour of day 0–23"),
                 "month":            ("",1,12,6,"1=Jan … 12=Dec"),
                 "hour_sin":         ("",-1,1,0,"sin(2π×hour/24) — auto-computed"),
                 "hour_cos":         ("",-1,1,1,"cos(2π×hour/24) — auto-computed"),
                 "lag_1":            ("",0,999,7,"PM2.5 1 hour ago µg/m³"),
                 "lag_3":            ("",0,999,7,"PM2.5 3 hours ago µg/m³"),
                 "lag_6":            ("",0,999,7,"PM2.5 6 hours ago µg/m³"),
                 "lag_12":           ("",0,999,6.5,"PM2.5 12 hours ago µg/m³"),
                 "roll_3":           ("",0,999,7,"3-hour rolling mean µg/m³"),
                 "roll_6":           ("",0,999,7,"6-hour rolling mean µg/m³"),
                 "roll_12":          ("",0,999,6.8,"12-hour rolling mean µg/m³"),
                 "roll_24":          ("",0,999,6.5,"24-hour rolling meandirectly comparable to WHO 2021 15 µg/m³ guideline"),
             }.items()]
    st.dataframe(pd.DataFrame(guide), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Day of Week Note</div>', unsafe_allow_html=True)
    st.markdown("The **Day of week** selector is for display context only — it is not passed to the model "
                "because day-of-week was not a training feature in the pipeline.")

    st.markdown("---")
    st.markdown('<div class="section-title">Month Name → Number</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"Month":k,"Value":v} for k,v in MONTH_TO_NUM.items()]),
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Alert Target</div>', unsafe_allow_html=True)
    st.latex(r"\text{alert} = \begin{cases}1 & y > 15\,\mu\text{g/m}^3\\0&\text{otherwise}\end{cases}")
    st.markdown("---")
    st.markdown('<div class="section-title">AQI Bands</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(AQI_BANDS, columns=["Lower","Upper","Label","Colour"]).drop(columns=["Colour"]),
        use_container_width=True, hide_index=True)
