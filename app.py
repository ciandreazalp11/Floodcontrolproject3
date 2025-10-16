# app.py
"""
Weather vs Flood Comparative Streamlit App (fixed)
- Accepts CSV/XLSX for Weather and Flood datasets
- Two uploads (weather + flood)
- Shared preprocessing
- Stores DataFrames in session_state (fixes UploadedFile errors)
- Comparison graphs and safe checks
"""

import os
import warnings
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# Optional forecasting libs
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SARIMAX_AVAILABLE = True
except Exception:
    SARIMAX_AVAILABLE = False

# --------------------
# Helpers
# --------------------
@st.cache_data
def read_table(file) -> pd.DataFrame:
    """
    Read uploaded CSV/XLSX into DataFrame. Accepts Streamlit UploadedFile or path-like.
    Always returns a pandas DataFrame.
    """
    fname = getattr(file, "name", None)
    # If file is a path string
    if isinstance(file, str):
        fname = file
    if fname is None:
        # fallback: try pandas autodetect
        return pd.read_csv(file)
    f_lower = fname.lower()
    try:
        if f_lower.endswith(".csv") or f_lower.endswith(".txt"):
            if isinstance(file, str):
                return pd.read_csv(file, encoding="latin1", low_memory=False)
            else:
                file.seek(0)
                return pd.read_csv(file, encoding="latin1", low_memory=False)
        elif f_lower.endswith(".xlsx") or f_lower.endswith(".xls"):
            if isinstance(file, str):
                return pd.read_excel(file)
            else:
                file.seek(0)
                return pd.read_excel(file)
        else:
            # try csv then excel
            try:
                if isinstance(file, str):
                    return pd.read_csv(file, encoding="latin1", low_memory=False)
                else:
                    file.seek(0)
                    return pd.read_csv(file, encoding="latin1", low_memory=False)
            except Exception:
                if isinstance(file, str):
                    return pd.read_excel(file)
                else:
                    file.seek(0)
                    return pd.read_excel(file)
    except Exception as e:
        raise e

def find_col(keywords, cols):
    """
    Return first column name that contains any keyword (case-insensitive)
    """
    cols_lower = [c.lower() for c in cols]
    for k in keywords:
        for i, c in enumerate(cols_lower):
            if k in c:
                return cols[i]
    return None

def preprocess_df(df: pd.DataFrame, date_col=None, main_numeric_col=None, area_col=None, damage_candidates=None):
    """
    Shared preprocessing logic for both weather & flood datasets.
    Returns processed df and the columns used.
    """
    df = df.copy()
    cols = df.columns.tolist()

    # auto-detect columns if not provided
    if date_col is None or date_col == "None":
        date_col = find_col(['date', 'datetime', 'time', 'day'], cols)
    if main_numeric_col is None or main_numeric_col == "None":
        numeric_candidates = ['water', 'level', 'wl', 'depth', 'height', 'rain', 'precip', 'temp', 'temperature', 'humid', 'rainfall']
        main_numeric_col = find_col(numeric_candidates, cols)
    if area_col is None or area_col == "None":
        area_col = find_col(['barangay', 'brgy', 'area', 'location', 'sitio', 'station'], cols)

    # Combine Date/Day/Year if exists
    if 'Date' in df.columns and 'Day' in df.columns and 'Year' in df.columns:
        df['__combined_date'] = df['Date'].astype(str) + ' ' + df['Day'].astype(str) + ', ' + df['Year'].astype(str)
        date_col = '__combined_date'
    # Fallback date index
    if date_col is None:
        df['__date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
        date_col = '__date'

    # parse date & set index
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
    df.index = pd.DatetimeIndex(df[date_col])

    # main numeric fallback: first numeric column
    if main_numeric_col is None or main_numeric_col not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found; please select a numeric column (e.g., rainfall, water level).")
        main_numeric_col = numeric_cols[0]

    # coerce to numeric & interpolate
    df[main_numeric_col] = pd.to_numeric(df[main_numeric_col], errors='coerce')
    df[main_numeric_col] = df[main_numeric_col].interpolate(method='linear', limit_direction='both')

    # z-score event detection
    df['zscore_main'] = stats.zscore(df[main_numeric_col].fillna(df[main_numeric_col].mean()))
    df['is_outlier_main'] = df['zscore_main'].abs() > 3

    # derived 'event' flag - for flood file this may indicate flood; for weather may indicate heavy rainfall/temp spikes
    occurrence_col = find_col(['flood', 'event', 'is_flood', 'flooded', 'occurrence'], cols)
    if occurrence_col and occurrence_col in df.columns:
        try:
            df['is_event'] = df[occurrence_col].astype(bool)
        except Exception:
            df['is_event'] = df[occurrence_col].notnull()
    else:
        threshold = df[main_numeric_col].mean() + 1.0 * df[main_numeric_col].std()
        df['is_event'] = (df[main_numeric_col] >= threshold) | (df['zscore_main'].abs() > 1.5)

    df['year'] = df.index.year
    df['month'] = df.index.to_period('M').astype(str)

    # damage columns (if present)
    damage_cols = []
    if damage_candidates is None:
        damage_candidates = ['infrastruct', 'infra', 'building', 'agri', 'agriculture', 'crop', 'farm', 'damage', 'loss', 'estimated_damage', 'total_damage']
    for k in damage_candidates:
        found = find_col([k], cols)
        if found and found not in damage_cols:
            damage_cols.append(found)
    for c in damage_cols:
        try:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')
        except Exception:
            pass

    return df, date_col, main_numeric_col, area_col, damage_cols

def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# Plot helpers
def plot_series(df, col, title=None, highlight_events=True):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df[col], label=col)
    if highlight_events and 'is_event' in df.columns:
        ax.scatter(df.index[df['is_event']], df[col][df['is_event']], s=18, label='Events')
    ax.set_title(title or f'{col} time series')
    ax.set_xlabel('Date'); ax.set_ylabel(col)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_bar_from_series(series, title, xlabel='period', ylabel='value'):
    fig, ax = plt.subplots(figsize=(9,4))
    series.plot(kind='bar', ax=ax)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig

# Forecasting (same as previous)
def run_sarima(series, train_frac=0.8, seasonal_period=12):
    if not SARIMAX_AVAILABLE:
        raise RuntimeError("SARIMAX (statsmodels) and sklearn required for forecasting.")
    series_m = series.resample('M').mean().fillna(0)
    if len(series_m) < seasonal_period:
        raise ValueError(f"Need at least {seasonal_period} monthly points for SARIMA.")
    split = int(len(series_m) * float(train_frac))
    train = series_m.iloc[:split]
    test = series_m.iloc[split:]
    best_aic = np.inf; best_res=None; best_order=None
    p_range = d_range = q_range = range(0,2)
    P_range = D_range = Q_range = range(0,2)
    for p in p_range:
        for d in d_range:
            for q in q_range:
                for P in P_range:
                    for D in D_range:
                        for Q in Q_range:
                            try:
                                mod = SARIMAX(train, order=(p,d,q), seasonal_order=(P,D,Q,seasonal_period),
                                              enforce_stationarity=False, enforce_invertibility=False)
                                res = mod.fit(disp=False)
                                if res.aic < best_aic:
                                    best_aic = res.aic; best_res = res; best_order = ((p,d,q),(P,D,Q,seasonal_period))
                            except Exception:
                                continue
    if best_res is None:
        raise RuntimeError("No SARIMA model fit successfully.")
    pred = best_res.get_prediction(start=test.index[0], end=test.index[-1], dynamic=False)
    forecast = pred.predicted_mean
    mae = mean_absolute_error(test, forecast); mse = mean_squared_error(test, forecast)
    return {'model':best_res, 'order':best_order, 'aic':best_aic, 'train':train, 'test':test, 'forecast':forecast, 'mae':mae, 'mse':mse}

# --------------------
# Streamlit App UI (kept original layout)
# --------------------
st.set_page_config(page_title="Weather vs Flood Analysis", layout="wide")
st.title("🌦️ Weather vs Flood — Comparative Analysis")

# Sidebar - navigation & uploads
st.sidebar.header("1) Upload datasets")
weather_file = st.sidebar.file_uploader("Upload Weather dataset (CSV or XLSX)", type=['csv','txt','xlsx','xls'], key='weather')
flood_file = st.sidebar.file_uploader("Upload Flood dataset (CSV or XLSX)", type=['csv','txt','xlsx','xls'], key='flood')

st.sidebar.markdown("---")
st.sidebar.header("2) Options")
use_example = st.sidebar.checkbox("Use example CSVs in repo if available", value=False)
show_raw = st.sidebar.checkbox("Show raw previews on preview page", value=False)

st.sidebar.markdown("---")
st.sidebar.header("Status")
# show simple status badges
if weather_file is not None or (use_example and os.path.exists('WEATHER DATASET.xlsx')):
    st.sidebar.success("Weather: uploaded")
else:
    st.sidebar.info("Weather: not uploaded")
if flood_file is not None or (use_example and os.path.exists('cleaned_flood_data.csv')):
    st.sidebar.success("Flood: uploaded")
else:
    st.sidebar.info("Flood: not uploaded")

st.sidebar.markdown("---")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Upload & Preview", "Preprocessing", "Comparison & Analysis", "Forecasting (Weather)", "Summary & Download"])

# Output folder for images
OUTDIR = 'flood_weather_outputs'
os.makedirs(OUTDIR, exist_ok=True)

# Session state initialization (store DataFrames, not UploadedFile)
if 'weather_raw' not in st.session_state: st.session_state.weather_raw = None
if 'flood_raw' not in st.session_state: st.session_state.flood_raw = None
if 'weather' not in st.session_state: st.session_state.weather = None
if 'flood' not in st.session_state: st.session_state.flood = None
if 'weather_main' not in st.session_state: st.session_state.weather_main = None
if 'flood_main' not in st.session_state: st.session_state.flood_main = None
if 'weather_date' not in st.session_state: st.session_state.weather_date = None
if 'flood_date' not in st.session_state: st.session_state.flood_date = None
if 'weather_area' not in st.session_state: st.session_state.weather_area = None
if 'flood_area' not in st.session_state: st.session_state.flood_area = None
if 'weather_damage_cols' not in st.session_state: st.session_state.weather_damage_cols = []
if 'flood_damage_cols' not in st.session_state: st.session_state.flood_damage_cols = []

# Load example if requested and files not provided
if use_example and (weather_file is None and flood_file is None):
    w_path = 'WEATHER DATASET.xlsx'  # example filename from earlier uploads if present
    f_path = 'cleaned_flood_data.csv'
    if os.path.exists(w_path):
        try:
            st.session_state.weather_raw = read_table(w_path)
            st.sidebar.success("Loaded example weather file")
        except Exception:
            st.sidebar.warning("Example weather file not found or failed to read")
    if os.path.exists(f_path):
        try:
            st.session_state.flood_raw = read_table(f_path)
            st.sidebar.success("Loaded example flood file")
        except Exception:
            st.sidebar.warning("Example flood file not found or failed to read")

# If user uploads, read into DataFrame and store in session_state.weather_raw / flood_raw
if weather_file is not None:
    try:
        dfw = read_table(weather_file)  # returns DataFrame
        st.session_state.weather_raw = dfw
        st.sidebar.success("Weather file read into DataFrame")
    except Exception as e:
        st.sidebar.error(f"Failed to read weather file: {e}")

if flood_file is not None:
    try:
        dff = read_table(flood_file)
        st.session_state.flood_raw = dff
        st.sidebar.success("Flood file read into DataFrame")
    except Exception as e:
        st.sidebar.error(f"Failed to read flood file: {e}")

# --------------------
# Pages
# --------------------
# 1) Upload & Preview
if page == "Upload & Preview":
    st.header("1. Upload & Preview")
    st.write("Upload a Weather dataset and a Flood dataset (CSV or XLSX). The app will run the same preprocessing on both and then compare patterns.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Weather dataset (raw)")
        if st.session_state.weather_raw is None:
            st.info("No Weather dataset loaded. Upload via the sidebar or enable example.")
        else:
            # weather_raw is guaranteed to be a DataFrame here (we read it earlier)
            st.write("Shape:", st.session_state.weather_raw.shape)
            if show_raw:
                st.dataframe(st.session_state.weather_raw.head(200))
            st.download_button("Download Weather raw CSV", data=to_csv_bytes(st.session_state.weather_raw), file_name="weather_raw.csv")
    with col2:
        st.subheader("Flood dataset (raw)")
        if st.session_state.flood_raw is None:
            st.info("No Flood dataset loaded. Upload via the sidebar or enable example.")
        else:
            st.write("Shape:", st.session_state.flood_raw.shape)
            if show_raw:
                st.dataframe(st.session_state.flood_raw.head(200))
            st.download_button("Download Flood raw CSV", data=to_csv_bytes(st.session_state.flood_raw), file_name="flood_raw.csv")

# 2) Preprocessing
elif page == "Preprocessing":
    st.header("2. Preprocessing")
    if st.session_state.weather_raw is None and st.session_state.flood_raw is None:
        st.warning("Please upload at least one dataset on the sidebar.")
    else:
        # Weather block
        st.subheader("Weather preprocessing")
        if st.session_state.weather_raw is None:
            st.info("No weather dataset uploaded.")
        else:
            w_df = st.session_state.weather_raw.copy()
            w_cols = w_df.columns.tolist()
            st.write("Detected columns:", w_cols[:10], "..." if len(w_cols)>10 else "")
            auto_w_date = find_col(['date','datetime','time','day'], w_cols)
            auto_w_main = find_col(['rain','precip','temp','temperature','humid','rainfall'], w_cols)
            auto_w_area = find_col(['station','location','area','site'], w_cols)
            st.write("Auto-detected: date:", auto_w_date, "| main:", auto_w_main, "| area:", auto_w_area)
            w_date_choice = st.selectbox("Weather: Date column", options=[None]+w_cols, index=(w_cols.index(auto_w_date)+1 if auto_w_date in w_cols else 0), key='w_date')
            w_main_choice = st.selectbox("Weather: Main numeric (rain/temp/etc)", options=[None]+w_cols, index=(w_cols.index(auto_w_main)+1 if auto_w_main in w_cols else 0), key='w_main')
            w_area_choice = st.selectbox("Weather: Area/Station column (optional)", options=[None]+w_cols, index=(w_cols.index(auto_w_area)+1 if auto_w_area in w_cols else 0), key='w_area')
            if st.button("Run Weather Preprocessing"):
                try:
                    w_proc, w_date_used, w_main_used, w_area_used, w_damage_cols = preprocess_df(w_df, date_col=(w_date_choice if w_date_choice is not None else None),
                                                                                                    main_numeric_col=(w_main_choice if w_main_choice is not None else None),
                                                                                                    area_col=(w_area_choice if w_area_choice is not None else None))
                    st.session_state.weather = w_proc
                    st.session_state.weather_date = w_date_used
                    st.session_state.weather_main = w_main_used
                    st.session_state.weather_area = w_area_used
                    st.session_state.weather_damage_cols = w_damage_cols
                    st.success("Weather preprocessing done")
                    st.write("Used date:", w_date_used, "Used main numeric:", w_main_used)
                    # show a small sample
                    cols_to_show = [c for c in [w_date_used, w_main_used, 'is_event', 'zscore_main'] if c in w_proc.columns]
                    st.dataframe(w_proc[cols_to_show].head(200))
                except Exception as e:
                    st.error(f"Weather preprocessing failed: {e}")

        st.markdown("---")
        # Flood block
        st.subheader("Flood preprocessing")
        if st.session_state.flood_raw is None:
            st.info("No flood dataset uploaded.")
        else:
            f_df = st.session_state.flood_raw.copy()
            f_cols = f_df.columns.tolist()
            st.write("Detected columns:", f_cols[:10], "..." if len(f_cols)>10 else "")
            auto_f_date = find_col(['date','datetime','time','day'], f_cols)
            auto_f_main = find_col(['water','level','wl','depth','height'], f_cols)
            auto_f_area = find_col(['barangay','brgy','area','location','sitio'], f_cols)
            st.write("Auto-detected: date:", auto_f_date, "| main:", auto_f_main, "| area:", auto_f_area)
            f_date_choice = st.selectbox("Flood: Date column", options=[None]+f_cols, index=(f_cols.index(auto_f_date)+1 if auto_f_date in f_cols else 0), key='f_date')
            f_main_choice = st.selectbox("Flood: Main numeric (water level)", options=[None]+f_cols, index=(f_cols.index(auto_f_main)+1 if auto_f_main in f_cols else 0), key='f_main')
            f_area_choice = st.selectbox("Flood: Area column (optional)", options=[None]+f_cols, index=(f_cols.index(auto_f_area)+1 if auto_f_area in f_cols else 0), key='f_area')
            if st.button("Run Flood Preprocessing"):
                try:
                    f_proc, f_date_used, f_main_used, f_area_used, f_damage_cols = preprocess_df(f_df, date_col=(f_date_choice if f_date_choice is not None else None),
                                                                                                    main_numeric_col=(f_main_choice if f_main_choice is not None else None),
                                                                                                    area_col=(f_area_choice if f_area_choice is not None else None))
                    st.session_state.flood = f_proc
                    st.session_state.flood_date = f_date_used
                    st.session_state.flood_main = f_main_used
                    st.session_state.flood_area = f_area_used
                    st.session_state.flood_damage_cols = f_damage_cols
                    st.success("Flood preprocessing done")
                    st.write("Used date:", f_date_used, "Used main numeric:", f_main_used)
                    cols_to_show = [c for c in [f_date_used, f_main_used, 'is_event', 'zscore_main'] if c in f_proc.columns]
                    st.dataframe(f_proc[cols_to_show].head(200))
                except Exception as e:
                    st.error(f"Flood preprocessing failed: {e}")

# 3) Comparison & Analysis
elif page == "Comparison & Analysis":
    st.header("3. Comparison & Analysis")
    # REQUIRE both cleaned datasets to compare
    if 'weather' not in st.session_state or st.session_state.weather is None:
        st.warning("Please preprocess the Weather dataset first (Preprocessing page).")
        st.stop()
    if 'flood' not in st.session_state or st.session_state.flood is None:
        st.warning("Please preprocess the Flood dataset first (Preprocessing page).")
        st.stop()

    # Safe to access now (DataFrames)
    w = st.session_state.weather.copy()
    f = st.session_state.flood.copy()
    wmain = st.session_state.weather_main
    fmain = st.session_state.flood_main

    st.subheader("A) Time series side-by-side")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Weather time series")
        try:
            figw = plot_series(w, wmain, title=f"Weather: {wmain}", highlight_events=True)
            st.pyplot(figw)
            figw.savefig(os.path.join(OUTDIR, "weather_timeseries.png"))
        except Exception as e:
            st.error(f"Failed plotting weather series: {e}")
    with col2:
        st.write("Flood time series")
        try:
            figf = plot_series(f, fmain, title=f"Flood: {fmain}", highlight_events=True)
            st.pyplot(figf)
            figf.savefig(os.path.join(OUTDIR, "flood_timeseries.png"))
        except Exception as e:
            st.error(f"Failed plotting flood series: {e}")

    st.subheader("B) Monthly aggregation & correlation (safe alignment)")
    try:
        w_month = w.resample('M')[wmain].mean().rename('weather_mean')
        f_month = f.resample('M')[fmain].mean().rename('flood_mean')
        common_idx = w_month.index.intersection(f_month.index)
        if len(common_idx) == 0:
            st.info("No overlapping months between datasets to compare. Ensure date ranges overlap.")
        else:
            w_month_a = w_month.loc[common_idx]
            f_month_a = f_month.loc[common_idx]
            df_pair = pd.DataFrame({'weather':w_month_a.values, 'flood':f_month_a.values}, index=common_idx)
            st.write("Monthly aggregated sample:")
            st.dataframe(df_pair.head(50))
            corr = df_pair['weather'].corr(df_pair['flood'])
            st.metric("Monthly correlation (weather vs flood)", f"{corr:.3f}")
            st.write("Scatter: weather vs flood (monthly mean)")
            fig_sc, ax = plt.subplots(figsize=(6,5))
            ax.scatter(df_pair['weather'], df_pair['flood'])
            ax.set_xlabel('Weather (monthly mean)')
            ax.set_ylabel('Flood (monthly mean)')
            ax.set_title(f"Correlation = {corr:.3f}")
            plt.tight_layout()
            st.pyplot(fig_sc)
            fig_sc.savefig(os.path.join(OUTDIR, "weather_vs_flood_scatter.png"))
    except Exception as e:
        st.error(f"Monthly comparison failed: {e}")

    st.subheader("C) Yearly summary comparison (safe)")
    try:
        w_year = w.groupby('year')[wmain].mean()
        f_year = f.groupby('year')[fmain].mean()
        df_year = pd.DataFrame({'weather_avg': w_year, 'flood_avg': f_year}).dropna()
        if df_year.empty:
            st.info("No overlapping yearly data.")
        else:
            fig_y = plot_bar_from_series(df_year['weather_avg'], "Weather Avg per Year", xlabel='Year', ylabel='Weather Avg')
            st.pyplot(fig_y)
            fig_yy = plot_bar_from_series(df_year['flood_avg'], "Flood Avg per Year", xlabel='Year', ylabel='Flood Avg')
            st.pyplot(fig_yy)
            st.write("Yearly correlation:", df_year['weather_avg'].corr(df_year['flood_avg']))
    except Exception as e:
        st.error(f"Yearly comparison failed: {e}")

# 4) Forecasting
elif page == "Forecasting (Weather)":
    st.header("4. Forecasting (Weather-driven)")
    if 'weather' not in st.session_state or st.session_state.weather is None:
        st.warning("Please preprocess the Weather dataset first (Preprocessing page).")
        st.stop()
    if not SARIMAX_AVAILABLE:
        st.error("Forecasting requires 'statsmodels' and 'scikit-learn'. Install them or add to requirements.")
        st.info("Local install: pip install statsmodels scikit-learn")
        st.stop()

    w = st.session_state.weather.copy()
    wmain = st.session_state.weather_main
    st.write("Forecasting on:", wmain)
    train_frac = st.slider("Training fraction for SARIMA", 0.5, 0.95, 0.8)
    run_fore = st.button("Run Weather SARIMA Forecast")
    if run_fore:
        try:
            with st.spinner("Running SARIMA grid search (short grid)..."):
                res = run_sarima(w[wmain], train_frac=train_frac, seasonal_period=12)
            st.success("SARIMA completed")
            st.write("Best order:", res['order'])
            st.write("AIC:", float(res['aic']))
            st.write("MAE:", float(res['mae']), "MSE:", float(res['mse']))
            # Plot results
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(res['train'].index, res['train'], label='Train')
            ax.plot(res['test'].index, res['test'], label='Test')
            ax.plot(res['forecast'].index, res['forecast'], label='Forecast')
            ax.set_title("SARIMA: Actual vs Forecast (monthly aggregated)")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            fig.savefig(os.path.join(OUTDIR, "weather_sarima_forecast.png"))
            # Risk flagging (only if flood preprocessed)
            if 'flood' in st.session_state and st.session_state.flood is not None:
                f = st.session_state.flood
                try:
                    w_month = w.resample('M')[wmain].mean().fillna(0)
                    f_events_month = f.resample('M')['is_event'].sum()
                    flood_months = f_events_month[f_events_month > 0].index
                    if len(flood_months) > 0:
                        weather_when_floods = w_month.reindex(flood_months).dropna()
                        if not weather_when_floods.empty:
                            risk_threshold = weather_when_floods.median()
                            st.write("Estimated weather threshold (median weather value on flood months):", float(risk_threshold))
                            forecast_monthly = res['forecast']
                            risk_months = forecast_monthly[forecast_monthly >= float(risk_threshold)]
                            if not risk_months.empty:
                                st.warning(f"⚠️ Forecast indicates {len(risk_months)} monthly periods where weather >= flood-associated threshold.")
                                st.dataframe(risk_months.reset_index().rename(columns={0:'forecast_value'}))
                            else:
                                st.info("No forecasted months exceed the flood-associated weather threshold.")
                        else:
                            st.info("No weather values for flood months to compute threshold.")
                    else:
                        st.info("Flood dataset has no monthly events; cannot infer weather threshold for flood risk.")
                except Exception as e:
                    st.error(f"Risk flagging failed: {e}")
        except Exception as e:
            st.error(f"Forecast failed: {e}")

# 5) Summary & Download
elif page == "Summary & Download":
    st.header("5. Summary & Download")
    if 'weather' not in st.session_state and 'flood' not in st.session_state:
        st.info("Nothing to summarize. Upload and preprocess datasets first.")
    else:
        if 'weather' in st.session_state and st.session_state.weather is not None:
            st.subheader("Weather cleaned dataset")
            st.write("Shape:", st.session_state.weather.shape)
            st.dataframe(st.session_state.weather.head(200))
            st.download_button("Download cleaned weather CSV", data=to_csv_bytes(st.session_state.weather.reset_index()), file_name="weather_cleaned.csv")
        if 'flood' in st.session_state and st.session_state.flood is not None:
            st.subheader("Flood cleaned dataset")
            st.write("Shape:", st.session_state.flood.shape)
            st.dataframe(st.session_state.flood.head(200))
            st.download_button("Download cleaned flood CSV", data=to_csv_bytes(st.session_state.flood.reset_index()), file_name="flood_cleaned.csv")

        # Yearly summary
        w_summary = pd.DataFrame(); f_summary = pd.DataFrame()
        if 'weather' in st.session_state and st.session_state.weather is not None:
            w = st.session_state.weather; wmain = st.session_state.weather_main
            w_summary = w.groupby('year')[wmain].mean().reset_index().rename(columns={wmain:'weather_avg'})
        if 'flood' in st.session_state and st.session_state.flood is not None:
            f = st.session_state.flood; fmain = st.session_state.flood_main
            f_summary = f.groupby('year')[fmain].mean().reset_index().rename(columns={fmain:'flood_avg'})

        if not w_summary.empty or not f_summary.empty:
            summary = pd.merge(w_summary, f_summary, on='year', how='outer').sort_values('year')
            st.subheader("Yearly summary")
            st.dataframe(summary)
            st.download_button("Download yearly summary CSV", data=to_csv_bytes(summary), file_name="yearly_summary.csv")

        st.write("Saved output files (images) in folder:", OUTDIR)
        st.write(os.listdir(OUTDIR))

st.sidebar.markdown("---")
st.sidebar.write("App flow: Upload -> Preprocess -> Compare -> Forecast -> Export. If auto-detection misfires, manually select columns during Preprocessing.")
