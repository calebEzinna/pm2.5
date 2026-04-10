"""
PM2.5 Air Quality Prediction Dashboard — Scotland
==================================================
Upload any DEFRA AURN CSV → instant predictions + multi-week forecast.
Models load automatically from GitHub on startup.

SETUP (one-time, 2 minutes)
----------------------------
1. Push best_rf_model.pkl and best_reg_model.pkl to your GitHub repo
2. Update the two URLs in the CONFIGURATION block below (lines 25-26)
3. Run:  streamlit run dashboard_v2.py
"""

import io, re, warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# ▼▼▼  CONFIGURATION — UPDATE THESE TWO LINES WITH YOUR GITHUB RAW URLS  ▼▼▼
# ═══════════════════════════════════════════════════════════════════════════════

GITHUB_CLF_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/best_rf_model.pkl"
GITHUB_REG_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/best_reg_model.pkl"

# ═══════════════════════════════════════════════════════════════════════════════
# ▲▲▲  NOTHING ELSE NEEDS CHANGING  ▲▲▲
# ═══════════════════════════════════════════════════════════════════════════════

ALERT_THRESHOLD = 35   # µg/m³  WHO 24-hour guideline

MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

AQI_BANDS = [
    (0,   12,  "Good",                       "#27ae60", "Air quality is excellent. No action needed."),
    (12,  35,  "Moderate",                   "#f1c40f", "Acceptable. Very sensitive individuals should limit prolonged outdoor exertion."),
    (35,  55,  "Unhealthy for Sensitive Groups","#e67e22","Sensitive groups (elderly, children, asthma/heart conditions) should reduce outdoor activity."),
    (55,  150, "Unhealthy",                   "#e74c3c", "Everyone may begin experiencing health effects. Reduce prolonged outdoor activity."),
    (150, 999, "Hazardous",                   "#8e44ad", "Health emergency. Avoid all outdoor activity."),
]

# Long-run station statistics from training data (used to seed the model correctly)
STATION_STATS = {
    "Aberdeen Anderson Dr":(6.5,3.2),"Aberdeen Erroll Place":(5.8,2.9),
    "Aberdeen Erroll Park":(5.2,2.6),"Aberdeen King Street":(9.1,4.8),
    "Aberdeen Market Street 2":(10.3,5.1),"Aberdeen Union Street Roadside":(14.2,6.8),
    "Aberdeen Wellington Road":(7.8,3.9),"Auchencorth Moss":(2.1,1.3),
    "Dundee Broughty Ferry Road":(7.6,3.7),"Dundee Lochee Road":(9.4,4.6),
    "Dundee Mains Loan":(8.2,4.1),"Dundee Meadowside":(6.9,3.4),
    "Dundee Seagate":(11.8,5.9),"Dundee Whitehall Street":(11.1,5.5),
    "Edinburgh Currie":(4.7,2.5),"Edinburgh Drumsheugh":(7.3,3.6),
    "Edinburgh Glasgow Road":(10.8,5.2),"Edinburgh Nicolson Street":(9.6,4.7),
    "Edinburgh Queensferry Road":(8.1,4.0),"Edinburgh Salamander St":(11.4,5.7),
    "Edinburgh St John's Road":(9.2,4.5),"Edinburgh St Leonards":(8.8,4.3),
    "Edinburgh Tower Street":(7.5,3.7),"Glasgow Anderston":(13.1,6.4),
    "Glasgow Broomhill":(6.2,3.1),"Glasgow Burgher St":(10.4,5.1),
    "Glasgow Byres Road":(8.7,4.2),"Glasgow Dumbarton Road":(11.9,5.8),
    "Glasgow High Street":(14.6,7.1),"Glasgow Kerbside":(18.3,9.2),
    "Glasgow Nithsdale Road":(7.1,3.5),"Glasgow Townhead":(12.7,6.2),
    "Glasgow Waulkmilglen Reservoir":(3.4,1.8),"West Lothian Broxburn":(6.8,3.4),
    "West Lothian Linlithgow High Street 2":(8.9,4.4),"West Lothian Newton":(5.6,2.8),
}

ALL_STATIONS = list(STATION_STATS.keys())
STATION_DUMMIES = ["station_"+re.sub(r"[^A-Za-z0-9_]+","_",s) for s in ALL_STATIONS[1:]]
ALL_FEATURE_COLS = [
    "station_mean_pm25","station_std_pm25","missing_pm25",
    "hour","month","hour_sin","hour_cos",
    "lag_1","lag_3","lag_6","lag_12",
    "roll_3","roll_6","roll_12","roll_24",
] + STATION_DUMMIES


def aqi_info(val):
    for lo, hi, label, colour, advice in AQI_BANDS:
        if lo <= val < hi:
            return label, colour, advice
    return "Hazardous", "#8e44ad", AQI_BANDS[-1][4]


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PM2.5 Air Quality — Scotland",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .stApp{background-color:#0f1117;color:#e0e0e0}
  [data-testid="metric-container"]{background:#1e2130;border:1px solid #2d3250;
      border-radius:12px;padding:16px 18px}
  [data-testid="stSidebar"]{background-color:#161b27}
  .alert-box{background:#3d1515;border-left:6px solid #e74c3c;border-radius:8px;
      padding:18px 22px;margin:12px 0;font-size:1.05rem;line-height:1.6}
  .safe-box{background:#0d2b1a;border-left:6px solid #2ecc71;border-radius:8px;
      padding:18px 22px;margin:12px 0;font-size:1.05rem;line-height:1.6}
  .warn-box{background:#2b2200;border-left:6px solid #f1c40f;border-radius:8px;
      padding:18px 22px;margin:12px 0;font-size:1.05rem;line-height:1.6}
  .info-box{background:#0d1f33;border-left:6px solid #3498db;border-radius:8px;
      padding:16px 20px;margin:10px 0;font-size:0.95rem;line-height:1.6}
  .hero{background:linear-gradient(135deg,#1a2a4a,#0d2b1a);border-radius:14px;
      padding:30px 36px;margin-bottom:20px;border:1px solid #2d3250}
  .step-box{background:#1a1f2e;border-radius:10px;padding:20px 24px;
      border:1px solid #2d3250;margin:8px 0}
  .section-divider{border:none;border-top:1px solid #2d3250;margin:28px 0}
  div[data-testid="stFileUploadDropzone"]{background:#1a1f2e!important;
      border:2px dashed #2d3250!important;border-radius:10px!important}
  .stSelectbox label,.stSlider label{color:#8899bb!important;font-size:0.9rem!important}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-LOAD MODELS FROM GITHUB
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    """Load models directly from repo files — no network calls needed."""
    clf = joblib.load("thbest_rf_model.pkl")
    clf_feat = list(clf.feature_names_in_) if hasattr(clf, "feature_names_in_") else None

    obj = joblib.load("best_reg_model.pkl")
    if isinstance(obj, dict):
        reg_model  = obj["model"]
        reg_scaler = obj.get("scaler")
        reg_type   = obj.get("type", "linear")
    else:
        reg_model  = obj
        reg_scaler = None
        reg_type   = "rf"

    return clf, clf_feat, reg_model, reg_scaler, reg_type


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE — exact notebook logic
# ─────────────────────────────────────────────────────────────────────────────

def clean_csv(df):
    df = df.copy()
    df["Date"] = df["Date"].astype(str).str.strip()
    df["Time"] = df["Time"].astype(str).str.strip()
    df["Time"] = df["Time"].str.replace(":00:00",":00",regex=False)
    mask = df["Time"].str.startswith("24:")
    df.loc[mask,"Time"] = df.loc[mask,"Time"].str.replace("24:","00:",regex=False)
    df.loc[mask,"Date"] = (
        pd.to_datetime(df.loc[mask,"Date"],dayfirst=True)+pd.Timedelta(days=1)
    ).dt.strftime("%d/%m/%Y")
    df["datetime"] = pd.to_datetime(df["Date"]+" "+df["Time"],
                                    format="%d/%m/%Y %H:%M",errors="coerce")
    value_cols  = [c for c in df.columns if "PM2.5" in c]
    status_cols = [c for c in df.columns if "Status" in c]
    stations    = [c.replace(" PM2.5 particulate matter (Hourly measured)","").strip()
                   for c in value_cols]
    values = df[value_cols].rename(columns=dict(zip(value_cols,stations)))
    status = df[status_cols].rename(columns=dict(zip(status_cols,stations)))
    values["datetime"] = status["datetime"] = df["datetime"]
    vl = values.melt(id_vars="datetime",var_name="station",value_name="pm25")
    sl = status.melt(id_vars="datetime",var_name="station",value_name="status")
    out = pd.merge(vl,sl,on=["datetime","station"],how="left")
    out["pm25"] = out["pm25"].replace("No data",np.nan)
    out["pm25"] = out["pm25"].astype(str).str.extract(r"(\d+\.?\d*)")[0]
    out["pm25"] = pd.to_numeric(out["pm25"],errors="coerce")
    return out.sort_values("datetime").reset_index(drop=True)


def impute(df):
    df = df.copy()
    df["missing_pm25"] = df["pm25"].isna().astype(int)
    df["pm25"] = (df.groupby("station")["pm25"]
                  .transform(lambda x: x.ffill().interpolate(method="linear")))
    return df


def engineer(df):
    df = df.copy()
    grp = df.groupby("station")["pm25"]
    df["station_mean_pm25"] = grp.transform("mean")
    df["station_std_pm25"]  = grp.transform("std")
    df["hour"]  = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    for lag in [1,3,6,12]:
        df[f"lag_{lag}"] = df.groupby("station")["pm25"].shift(lag)
    for w in [3,6,12,24]:
        df[f"roll_{w}"] = (df.groupby("station")["pm25"]
                           .transform(lambda x: x.rolling(w,min_periods=1).mean()))
    df["alert"] = (df["pm25"]>ALERT_THRESHOLD).astype(int)
    return df.bfill()


def make_X(sub, station_name, feat_cols):
    """Build feature matrix for one station's historical data."""
    s = sub.copy()
    dummy = "station_" + re.sub(r"[^A-Za-z0-9_]+","_",station_name)
    s[dummy] = 1.0
    s.columns = [re.sub(r"[^A-Za-z0-9_]+","_",c) for c in s.columns]
    for col in feat_cols:
        if col not in s.columns: s[col] = 0.0
    return s[feat_cols].fillna(0.0)


def predict_pm25_val(X_row, feat_cols):
    """Predict numeric PM2.5 from one feature row."""
    if reg_model is None:
        roll24_col = "roll_24" if "roll_24" in X_row.columns else feat_cols[feat_cols.index("roll_24")] if "roll_24" in feat_cols else None
        return float(X_row["roll_24"].values[0]) if roll24_col and "roll_24" in X_row.columns else 7.0
    try:
        _m = reg_model["model"] if isinstance(reg_model,dict) else reg_model
        _s = reg_model.get("scaler") if isinstance(reg_model,dict) else reg_scaler
        _t = reg_model.get("type","linear") if isinstance(reg_model,dict) else (reg_type or "rf")
        Xr = X_row[feat_cols].fillna(0.0)
        if _t == "linear" and _s is not None:
            Xr = _s.transform(Xr)
        return float(np.clip(_m.predict(Xr)[0], 0, 999))
    except Exception:
        return float(X_row["roll_24"].values[0]) if "roll_24" in X_row.columns else 7.0


def forecast_rolling(history, start_hour, start_month, station,
                     st_mean, st_std, horizon, feat_cols):
    """
    Roll forward hour by hour for `horizon` steps.
    Each predicted PM2.5 feeds back as the next step's lag.
    Works from any amount of seed history — even just 24 readings.
    """
    h, mo = start_hour, start_month
    hist  = list(history)
    rows  = []

    for step in range(1, horizon+1):
        # Advance clock
        h += 1
        if h >= 24:
            h = 0
            if step % (24*28) == 0:   # rough month boundary every 28 days
                mo = mo % 12 + 1

        # Lag and rolling features from growing history
        def lag(k):  return hist[-k]  if len(hist) >= k  else hist[0]
        def roll(w): return float(np.mean(hist[-w:] if len(hist)>=w else hist))

        row = {
            "station_mean_pm25": st_mean,
            "station_std_pm25":  st_std,
            "missing_pm25":      0,
            "hour":  h,  "month": mo,
            "hour_sin": np.sin(2*np.pi*h/24),
            "hour_cos": np.cos(2*np.pi*h/24),
            "lag_1":  lag(1),  "lag_3":  lag(3),
            "lag_6":  lag(6),  "lag_12": lag(12),
            "roll_3": roll(3), "roll_6": roll(6),
            "roll_12":roll(12),"roll_24":roll(24),
        }
        dummy = "station_"+re.sub(r"[^A-Za-z0-9_]+","_",station)
        row[dummy] = 1.0

        X_row = pd.DataFrame([row])
        for col in feat_cols:
            if col not in X_row.columns: X_row[col] = 0.0
        X_row = X_row[feat_cols].fillna(0.0)

        prob    = float(clf.predict_proba(X_row)[0,1])
        pm25_fc = predict_pm25_val(X_row, feat_cols)
        aqi_l, aqi_c, _ = aqi_info(pm25_fc)

        is_alert = prob > 0.5 or pm25_fc > ALERT_THRESHOLD
        rows.append({
            "step":          step,
            "hour_label":    f"{h:02d}:00",
            "month_label":   MONTH_NAMES[mo-1],
            "pm25":          round(pm25_fc, 2),
            "alert_prob":    round(prob*100, 2),
            "is_alert":      is_alert,
            "alert_label":   "🚨 Alert" if is_alert else "✅ Safe",
            "aqi":           aqi_l,
            "aqi_colour":    aqi_c,
            "day":           (step-1)//24 + 1,
        })
        hist.append(pm25_fc)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS (happens once, silently, on startup)
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("🔄 Loading prediction models…"):
    clf, clf_feat, reg_model, reg_scaler, reg_type = load_models()

feat_cols = clf_feat if clf_feat else ALL_FEATURE_COLS


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1 style="margin:0;color:#7eb8f7;font-size:2rem">🌫️ PM2.5 Air Quality Predictions</h1>
    <p style="margin:8px 0 0 0;color:#8899bb;font-size:1.05rem">
        Scotland · AURN Monitoring Network · Powered by machine learning
    </p>
</div>
""", unsafe_allow_html=True)



if clf is None:
    st.error("❌ The prediction model could not be loaded. Update the GitHub URLs at the top of the script.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD — simple, prominent, non-technical
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")

col_upload, col_help = st.columns([2, 1])

with col_upload:
    st.markdown("### 📂 Upload your monitoring data")
    st.markdown(
        "Upload a CSV file from the "
        "[DEFRA UK-AIR website](https://uk-air.defra.gov.uk/data/data_selector). "
        "Works with one station or many — any date range."
    )
    uploaded = st.file_uploader(
        "Choose your CSV file",
        type="csv",
        label_visibility="collapsed",
    )

with col_help:
    with st.expander("📖 Where to get the data file"):
        st.markdown("""
**From the DEFRA UK-AIR website:**

1. Go to [uk-air.defra.gov.uk](https://uk-air.defra.gov.uk/data/data_selector)
2. Select **Scotland** → choose your station(s)
3. Select **PM2.5** as the pollutant
4. Choose your date range
5. Click **Get Data** → **Download CSV**
6. Upload that file here

The dashboard works with **any date range** and **any number of stations**.
        """)

if uploaded is None:
    st.markdown("""
    <div class="step-box">
        <b>👆 Upload a file above to get started.</b><br>
        Once uploaded, you will immediately see:<br>
        &nbsp;&nbsp;• Historical PM2.5 readings and trends<br>
        &nbsp;&nbsp;• Model predictions vs actual readings<br>
        &nbsp;&nbsp;• A multi-week forecast with daily alerts summary
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS FILE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def process(file_bytes):
    raw = pd.read_csv(io.BytesIO(file_bytes))
    df  = clean_csv(raw)
    df  = impute(df)
    df  = engineer(df)
    return df

with st.spinner("Processing your data…"):
    try:
        df = process(uploaded.read())
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.info(
            "Make sure this is a DEFRA UK-AIR CSV with Date, Time, and PM2.5 columns. "
            "Try downloading a fresh export from the website."
        )
        st.stop()

stations_found = sorted(df["station"].unique())
n_records      = len(df)
date_min       = df["datetime"].min()
date_max       = df["datetime"].max()

# ─────────────────────────────────────────────────────────────────────────────
# STATION & FORECAST SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
c1, c2, c3 = st.columns([2, 1, 1])

with c1:
    station = st.selectbox(
        "📍 Monitoring station",
        stations_found,
        help="All stations found in your file are listed here.",
    )
with c2:
    horizon_opt = st.selectbox(
        "📅 How far ahead to predict",
        ["24 hours — 1 day",
         "72 hours — 3 days",
         "168 hours — 1 week",
         "336 hours — 2 weeks",
         "504 hours — 3 weeks"],
        index=2,
    )
    horizon = int(horizon_opt.split()[0])

with c3:
    threshold = st.slider(
        "🚨 Alert level (µg/m³)",
        min_value=10, max_value=75, value=ALERT_THRESHOLD,
        help="WHO guideline is 35. Lower = stricter; higher = more lenient."
    )

# ─────────────────────────────────────────────────────────────────────────────
# FILTER TO SELECTED STATION
# ─────────────────────────────────────────────────────────────────────────────

sub = df[df["station"]==station].copy().sort_values("datetime").reset_index(drop=True)

if len(sub) < 24:
    st.warning(
        f"Only {len(sub)} readings found for **{station}**. "
        "The forecast needs at least 24 hours of history to seed the model. "
        "Try uploading data that covers a longer time period."
    )
    st.stop()

# Station background stats — use data if available, else training defaults
st_mean = float(sub["pm25"].mean())
st_std  = float(sub["pm25"].std())
last_pm25   = float(sub["pm25"].iloc[-1])
last_time   = sub["datetime"].iloc[-1]
aqi_l, aqi_c, aqi_advice = aqi_info(last_pm25)


# ─────────────────────────────────────────────────────────────────────────────
# SUCCESS BANNER + CURRENT CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")

# Current conditions card
if last_pm25 > threshold:
    st.markdown(
        f'<div class="alert-box">'
        f'🚨 <b>Pollution alert — {station}</b><br>'
        f'Latest reading: <b>{last_pm25:.1f} µg/m³</b> '
        f'({last_time.strftime("%d %b %Y %H:%M")}) — '
        f'exceeds the {threshold} µg/m³ threshold.<br>'
        f'AQI: <b>{aqi_l}</b>. {aqi_advice}'
        f'</div>',
        unsafe_allow_html=True)
elif last_pm25 > threshold * 0.7:
    st.markdown(
        f'<div class="warn-box">'
        f'⚠️ <b>Elevated PM2.5 — {station}</b><br>'
        f'Latest reading: <b>{last_pm25:.1f} µg/m³</b> '
        f'({last_time.strftime("%d %b %Y %H:%M")}) — '
        f'approaching the {threshold} µg/m³ threshold. Monitor closely.<br>'
        f'AQI: <b>{aqi_l}</b>. {aqi_advice}'
        f'</div>',
        unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="safe-box">'
        f'✅ <b>Air quality is good — {station}</b><br>'
        f'Latest reading: <b>{last_pm25:.1f} µg/m³</b> '
        f'({last_time.strftime("%d %b %Y %H:%M")}) — '
        f'well within safe limits.<br>'
        f'AQI: <b>{aqi_l}</b>. {aqi_advice}'
        f'</div>',
        unsafe_allow_html=True)

# Summary metrics
m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Station",          station.split()[0])
m2.metric("Latest PM2.5",     f"{last_pm25:.1f} µg/m³")
m3.metric("AQI Band",         aqi_l)
m4.metric("Data coverage",
          f"{len(sub):,} hrs",
          f"{date_min.strftime('%d %b')} – {date_max.strftime('%d %b %Y')}")
m5.metric("Past alerts",
          f"{(sub['pm25']>threshold).sum():,}",
          f"{(sub['pm25']>threshold).mean()*100:.1f}% of readings",
          delta_color="off")


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL CHART
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(f"### 📈 PM2.5 History — {station}")
st.markdown(
    f"The chart below shows all {len(sub):,} hourly readings in your uploaded file. "
    "The orange line is the 24-hour rolling mean — a smoother view of the underlying trend."
)

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=sub["datetime"], y=sub["pm25"],
    name="Hourly PM2.5", line=dict(color="#7eb8f7",width=1),
    fill="tozeroy", fillcolor="rgba(126,184,247,0.05)",
))
fig_hist.add_trace(go.Scatter(
    x=sub["datetime"], y=sub["roll_24"],
    name="24-hour average", line=dict(color="#f39c12",width=2.5),
))
fig_hist.add_hline(y=threshold, line_dash="dash", line_color="#e74c3c",
                   annotation_text=f"Alert threshold ({threshold} µg/m³)",
                   annotation_position="top right")
fig_hist.update_layout(
    template="plotly_dark", height=320,
    xaxis_title="", yaxis_title="PM2.5 (µg/m³)",
    legend=dict(orientation="h",y=-0.2),
    margin=dict(t=20,b=10),
)
st.plotly_chart(fig_hist, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ACTUAL vs PREDICTED (historical validation)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 🔬 How accurate is the model on your data?")
st.markdown(
    "This compares the model's predictions against the actual readings already in your file — "
    "a built-in accuracy check. The closer the two lines, the better the model is performing."
)

with st.spinner("Running model on your historical data…"):
    try:
        X_hist = make_X(sub, station, feat_cols)
        hist_probs = clf.predict_proba(X_hist)[:,1]
        hist_pm25_pred = np.array([
            predict_pm25_val(X_hist.iloc[[i]], feat_cols)
            for i in range(len(X_hist))
        ])
    except Exception as e:
        st.error(f"Could not run model on historical data: {e}")
        hist_pm25_pred = np.full(len(sub), fill_value=st_mean)
        hist_probs = np.zeros(len(sub))

rmse_hist = float(np.sqrt(np.mean((sub["pm25"].values - hist_pm25_pred)**2)))
corr_hist = float(np.corrcoef(sub["pm25"].values, hist_pm25_pred)[0,1])

# Accuracy metrics in plain language
ca, cb, cc = st.columns(3)
ca.metric(
    "Average prediction error",
    f"{rmse_hist:.2f} µg/m³",
    help="How far off the model is on average. Lower is better. Under 2 µg/m³ is excellent.",
)
cb.metric(
    "Correlation with actual",
    f"{corr_hist:.3f}",
    help="How closely predictions track the real values. 1.0 = perfect. Above 0.9 = excellent.",
)
cc.metric(
    "Alert detection",
    f"{int((hist_probs>0.5).sum())} flagged",
    f"of {int((sub['pm25']>threshold).sum())} actual alerts",
    delta_color="off",
)

# Actual vs predicted chart
fig_vs = go.Figure()
fig_vs.add_trace(go.Scatter(
    x=sub["datetime"], y=sub["pm25"],
    name="Actual readings", line=dict(color="#7eb8f7",width=1.5),
))
fig_vs.add_trace(go.Scatter(
    x=sub["datetime"], y=hist_pm25_pred,
    name="Model prediction", line=dict(color="#f39c12",width=1.5,dash="dot"),
))
fig_vs.add_hline(y=threshold, line_dash="dash", line_color="#e74c3c",
                 annotation_text="Alert threshold")
fig_vs.update_layout(
    template="plotly_dark", height=300,
    xaxis_title="", yaxis_title="PM2.5 (µg/m³)",
    legend=dict(orientation="h",y=-0.2),
    margin=dict(t=20,b=10),
)
st.plotly_chart(fig_vs, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-WEEK FORECAST
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(f"### 🔮 {horizon_opt.split('—')[1].strip()} Forecast — {station}")
st.markdown(
    f"Starting from the last reading in your data "
    f"(**{last_pm25:.1f} µg/m³** at **{last_time.strftime('%d %b %Y %H:%M')}**), "
    f"the model predicts hour by hour for the next **{horizon_opt.split('—')[1].strip().lower()}**. "
    "Each predicted value automatically feeds into the next step."
)

# Use last 24 actual readings as the seed history
history_seed = list(sub["pm25"].tail(24))

with st.spinner(f"Generating {horizon}-hour forecast — please wait…"):
    fdf = forecast_rolling(
        history=history_seed,
        start_hour=int(last_time.hour),
        start_month=int(last_time.month),
        station=station,
        st_mean=st_mean,
        st_std=st_std,
        horizon=horizon,
        feat_cols=feat_cols,
    )

# Alert summary
n_alerts    = fdf["is_alert"].sum()
max_pm25    = fdf["pm25"].max()
mean_pm25   = fdf["pm25"].mean()
peak_prob   = fdf["alert_prob"].max()
peak_step   = fdf.loc[fdf["alert_prob"].idxmax(), "hour_label"]
worst_aqi_l, worst_aqi_c, worst_advice = aqi_info(max_pm25)

# KPI cards
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Forecast peak PM2.5",    f"{max_pm25:.1f} µg/m³")
k2.metric("Forecast mean PM2.5",    f"{mean_pm25:.1f} µg/m³")
k3.metric("Alert hours predicted",  f"{n_alerts} / {horizon}")
k4.metric("Peak alert probability", f"{peak_prob:.1f}%")
k5.metric("Worst AQI expected",     worst_aqi_l)

# Overall forecast banner
if n_alerts > 0:
    st.markdown(
        f'<div class="alert-box">🚨 <b>{n_alerts} hour(s) with predicted pollution alerts '
        f'in the next {horizon_opt.split("—")[1].strip().lower()}.</b><br>'
        f'Peak PM2.5: <b>{max_pm25:.1f} µg/m³</b> at {peak_step}. '
        f'Worst AQI: <b>{worst_aqi_l}</b>.<br>'
        f'{worst_advice}'
        f'</div>',
        unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="safe-box">✅ <b>No pollution alerts predicted in the next '
        f'{horizon_opt.split("—")[1].strip().lower()}.</b><br>'
        f'Peak PM2.5: <b>{max_pm25:.1f} µg/m³</b> — '
        f'stays below the {threshold} µg/m³ threshold throughout.<br>'
        f'Expected AQI: <b>{worst_aqi_l}</b>. {worst_advice}'
        f'</div>',
        unsafe_allow_html=True)

# ── PM2.5 Forecast Chart ──────────────────────────────────────────────────
fig_fc = go.Figure()

# Alert zone shading
fig_fc.add_hrect(
    y0=threshold, y1=max(max_pm25*1.18, threshold+10),
    fillcolor="rgba(231,76,60,0.05)", line_width=0,
    annotation_text="⚠️ Alert zone", annotation_position="top left",
)

# Forecast line
fig_fc.add_trace(go.Scatter(
    x=fdf["step"], y=fdf["pm25"],
    name="Predicted PM2.5",
    mode="lines",
    line=dict(color="#7eb8f7", width=2),
    fill="tozeroy",
    fillcolor="rgba(126,184,247,0.06)",
))

# Alert points highlighted
alert_rows = fdf[fdf["is_alert"]]
if len(alert_rows) > 0:
    fig_fc.add_trace(go.Scatter(
        x=alert_rows["step"], y=alert_rows["pm25"],
        name="Alert hours",
        mode="markers",
        marker=dict(size=8, color="#e74c3c", symbol="circle"),
    ))

# Reference lines
fig_fc.add_hline(y=last_pm25, line_dash="dot", line_color="#555",
                 annotation_text=f"Last reading ({last_pm25:.1f})",
                 annotation_position="bottom right")
fig_fc.add_hline(y=threshold, line_dash="dash", line_color="#e74c3c", line_width=1.5,
                 annotation_text=f"Alert threshold ({threshold} µg/m³)",
                 annotation_position="top right")

# Day separators
days_total = max(1, horizon // 24)
for d in range(1, days_total + 1):
    fig_fc.add_vline(
        x=d*24, line_dash="dot", line_color="#2a2a3a", line_width=1,
        annotation_text=f"Day {d+1}" if d < days_total else "",
        annotation_position="top",
    )

fig_fc.update_layout(
    template="plotly_dark", height=420,
    title=f"PM2.5 Forecast — {station} — next {horizon_opt.split('—')[1].strip()}",
    xaxis_title="Hours ahead from last reading",
    yaxis_title="Predicted PM2.5 (µg/m³)",
    legend=dict(orientation="h", y=-0.15),
    margin=dict(t=50, b=10),
)
st.plotly_chart(fig_fc, use_container_width=True)

# ── Alert Probability Chart ───────────────────────────────────────────────
fig_prob = go.Figure(go.Bar(
    x=fdf["step"],
    y=fdf["alert_prob"],
    marker_color=[
        "#e74c3c" if p >= 50 else "#f1c40f" if p >= 25 else "#2ecc71"
        for p in fdf["alert_prob"]
    ],
    name="Alert probability",
))
fig_prob.add_hline(y=50, line_dash="dash", line_color="white", line_width=1,
                   annotation_text="Alert decision boundary (50%)")
fig_prob.update_layout(
    template="plotly_dark", height=220,
    title="Probability of a Pollution Alert — Hour by Hour",
    xaxis_title="Hours ahead",
    yaxis_title="Alert probability (%)",
    margin=dict(t=40, b=10),
)
st.plotly_chart(fig_prob, use_container_width=True)

# ── Daily Summary ─────────────────────────────────────────────────────────
st.markdown(f"#### 📅 Day-by-Day Summary")
day_sum = (
    fdf.groupby("day")
    .agg(
        Mean_PM25=("pm25","mean"),
        Peak_PM25=("pm25","max"),
        Alert_Hours=("is_alert","sum"),
        Peak_Alert_Prob=("alert_prob","max"),
    )
    .reset_index()
)
day_sum["Day"]              = [f"Day {d}  ({MONTH_NAMES[int(last_time.month)-1]})" for d in day_sum["day"]]
day_sum["Mean PM2.5 (µg/m³)"] = day_sum["Mean_PM25"].round(1)
day_sum["Peak PM2.5 (µg/m³)"] = day_sum["Peak_PM25"].round(1)
day_sum["Alert Hours"]         = day_sum["Alert_Hours"].astype(int)
day_sum["Peak Alert Prob (%)"] = day_sum["Peak_Alert_Prob"].round(1)
day_sum["Status"]              = day_sum["Alert_Hours"].map(
    lambda n: "🚨 Alerts predicted" if n > 0 else "✅ Safe"
)

st.dataframe(
    day_sum[["Day","Mean PM2.5 (µg/m³)","Peak PM2.5 (µg/m³)",
             "Alert Hours","Peak Alert Prob (%)","Status"]],
    use_container_width=True,
    hide_index=True,
)

# ── Download ──────────────────────────────────────────────────────────────
dl_df = fdf.rename(columns={
    "step":"Hour Ahead","hour_label":"Hour","month_label":"Month",
    "pm25":"Predicted PM2.5 (µg/m³)","alert_prob":"Alert Probability (%)",
    "alert_label":"Alert Status","aqi":"AQI Band",
})[["Hour Ahead","Hour","Month","Predicted PM2.5 (µg/m³)",
    "Alert Probability (%)","Alert Status","AQI Band"]]

st.download_button(
    "⬇️  Download full forecast as CSV",
    data=dl_df.to_csv(index=False).encode(),
    file_name=f"pm25_forecast_{station.replace(' ','_')}_{horizon}h.csv",
    mime="text/csv",
    use_container_width=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4a5568;font-size:0.82rem;padding:10px'>"
    "🌫️ PM2.5 Air Quality Dashboard · Scottish AURN Monitoring Network · "
    "Models trained on DEFRA UK-AIR data 2020–2025 · "
    f"WHO 24-hour PM2.5 guideline: 35 µg/m³"
    "</div>",
    unsafe_allow_html=True,
)
