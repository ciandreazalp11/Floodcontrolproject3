# app.py
"""
Complete Flood & Weather Analysis Streamlit App
- Upload CSV/XLSX for Weather & Flood
- Preprocessing with auto-detect columns
- Comparison, Forecasting, Summary & Download
- Fully fixed for session_state & widget conflicts
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

# Forecasting libraries
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SARIMAX_AVAILABLE = True
except Exception:
    SARIMAX_AVAILABLE = False

# --------------------
# Helper functions
# --------------------
@st.cache_data
def read_table(file) -> pd.DataFrame:
    fname = getattr(file, "name", None)
    if isinstance(file, str):
        fname = file
    if fname is None:
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
    cols_lower = [c.lower() for c in cols]
    for k in keywords:
        for i, c in enumerate(cols_lower):
            if k in c:
                return cols[i]
    return None

def preprocess_df(df: pd.DataFrame, date_col=None, main_numeric_col=None, area_col=None, damage_candidates=None):
    df = df.copy()
    cols = df.columns.tolist()

    if date_col is None or date_col == "None":
        date_col = find_col(['date', 'datetime', 'time', 'day'], cols)
    if main_numeric_col is None or main_numeric_col == "None":
        numeric_candidates = ['water', 'level', 'wl', 'depth', 'height', 'rain', 'precip', 'temp', 'temperature', 'humid', 'rainfall']
        main_numeric_col = find_col(numeric_candidates, cols)
    if area_col is None or area_col == "None":
        area_col = find_col(['barangay', 'brgy', 'area', 'location', 'sitio', 'station'], cols)

    if 'Date' in df.columns and 'Day' in df.columns and 'Year' in df.columns:
        df['__combined_date'] = df['Date'].astype(str) + ' ' + df['Day'].astype(str) + ', ' + df['Year'].astype(str)
        date_col = '__combined_date'

    if date_col is None:
        df['__date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
        date_col = '__date'

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
    df.index = pd.DatetimeIndex(df[date_col])

    if main_numeric_col is None or main_numeric_col not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found; please select a numeric column.")
        main_numeric_col = numeric_cols[0]

    df[main_numeric_col] = pd.to_numeric(df[main_numeric_col], errors='coerce')
    df[main_numeric_col] = df[main_numeric_col].interpolate(method='linear', limit_direction='both')

    df['zscore_main'] = stats.zscore(df[main_numeric_col].fillna(df[main_numeric_col].mean()))
    df['is_outlier_main'] = df['zscore_main'].abs() > 3

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

def run_sarima(series, train_frac=0.8, seasonal_period=12):
    if not SARIMAX_AVAILABLE:
        raise RuntimeError("SARIMAX and sklearn required for forecasting.")
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
# Streamlit App UI
# --------------------
st.set_page_config(page_title="Weather vs Flood Analysis", layout="wide")
st.title("🌦️ Weather vs Flood — Comparative Analysis")

# Sidebar - navigation & uploads
st.sidebar.header("1) Upload datasets")
weather_file = st.sidebar.file_uploader("Upload Weather dataset (CSV or XLSX)", type=['csv','txt','xlsx','xls'], key='weather_file')
flood_file   = st.sidebar.file_uploader("Upload Flood dataset (CSV or XLSX)", type=['csv','txt','xlsx','xls'], key='flood_file')

st.sidebar.markdown("---")
st.sidebar.header("2) Options")
show_raw    = st.sidebar.checkbox("Show raw previews", value=False)

st.sidebar.markdown("---")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Upload & Preview", "Preprocessing", "Comparison & Analysis", "Forecasting (Weather)", "Summary & Download"])

# Output folder
OUTDIR = 'flood_weather_outputs'
os.makedirs(OUTDIR, exist_ok=True)

# Initialize session_state
for key in ['weather_raw','flood_raw','weather_df','flood_df',
            'weather_main','flood_main','weather_date','flood_date',
            'weather_area','flood_area','weather_damage_cols','flood_damage_cols']:
    if key not in st.session_state:
        st.session_state[key] = None if 'cols' not in key else []

# Helper to safely get df
def get_df_safe(name):
    df = st.session_state.get(name, None)
    if df is None:
        st.warning(f"Dataset {name} not available. Please preprocess first.")
        st.stop()
    return df.copy()

# --------------------
# 1) Upload & Preview
# --------------------
if page == "Upload & Preview":
    st.header("1. Upload & Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Weather dataset (raw)")
        if st.session_state.weather_raw is None:
            st.info("No Weather dataset loaded.")
        else:
            st.write("Shape:", st.session_state.weather_raw.shape)
            if show_raw: st.dataframe(st.session_state.weather_raw.head(200))
            st.download_button("Download Weather raw CSV", data=to_csv_bytes(st.session_state.weather_raw), file_name="weather_raw.csv")
    with col2:
        st.subheader("Flood dataset (raw)")
        if st.session_state.flood_raw is None:
            st.info("No Flood dataset loaded.")
        else:
            st.write("Shape:", st.session_state.flood_raw.shape)
            if show_raw: st.dataframe(st.session_state.flood_raw.head(200))
            st.download_button("Download Flood raw CSV", data=to_csv_bytes(st.session_state.flood_raw), file_name="flood_raw.csv")

# --------------------
# 2) Preprocessing
# --------------------
elif page == "Preprocessing":
    st.header("2. Preprocessing")
    if weather_file:
        try:
            st.session_state.weather_raw = read_table(weather_file)
            st.success("Weather file loaded")
        except Exception as e:
            st.error(f"Failed to read weather file: {e}")
    if flood_file:
        try:
            st.session_state.flood_raw = read_table(flood_file)
            st.success("Flood file loaded")
        except Exception as e:
            st.error(f"Failed to read flood file: {e}")

    # Weather preprocessing
    if st.session_state.weather_raw is not None:
        st.subheader("Weather preprocessing")
        w_df = st.session_state.weather_raw.copy()
        w_cols = w_df.columns.tolist()
        auto_w_date = find_col(['date','datetime','time','day'], w_cols)
        auto_w_main = find_col(['rain','precip','temp','temperature','humid','rainfall'], w_cols)
        auto_w_area = find_col(['station','location','area','site'], w_cols)
        w_date_choice = st.selectbox("Weather: Date column", options=[None]+w_cols, index=(w_cols.index(auto_w_date)+1 if auto_w_date in w_cols else 0), key='w_date')
        w_main_choice = st.selectbox("Weather: Main numeric", options=[None]+w_cols, index=(w_cols.index(auto_w_main)+1 if auto_w_main in w_cols else 0), key='w_main')
        w_area_choice = st.selectbox("Weather: Area/Station column", options=[None]+w_cols, index=(w_cols.index(auto_w_area)+1 if auto_w_area in w_cols else 0), key='w_area')
        if st.button("Run Weather Preprocessing"):
            try:
                w_proc, w_date_used, w_main_used, w_area_used, w_damage_cols = preprocess_df(
                    w_df, date_col=w_date_choice, main_numeric_col=w_main_choice, area_col=w_area_choice)
                st.session_state.weather_df = w_proc
                st.session_state.weather_date = w_date_used
                st.session_state.weather_main = w_main_used
                st.session_state.weather_area = w_area_used
                st.session_state.weather_damage_cols = w_damage_cols
                st.success("Weather preprocessing done")
            except Exception as e:
                st.error(f"Weather preprocessing failed: {e}")

    # Flood preprocessing
    if st.session_state.flood_raw is not None:
        st.subheader("Flood preprocessing")
        f_df = st.session_state.flood_raw.copy()
        f_cols = f_df.columns.tolist()
        auto_f_date = find_col(['date','datetime','time','day'], f_cols)
        auto_f_main = find_col(['water','level','wl','depth','height'], f_cols)
        auto_f_area = find_col(['barangay','brgy','area','location','sitio'], f_cols)
        f_date_choice = st.selectbox("Flood: Date column", options=[None]+f_cols, index=(f_cols.index(auto_f_date)+1 if auto_f_date in f_cols else 0), key='f_date')
        f_main_choice = st.selectbox("Flood: Main numeric", options=[None]+f_cols, index=(f_cols.index(auto_f_main)+1 if auto_f_main in f_cols else 0), key='f_main')
        f_area_choice = st.selectbox("Flood: Area column", options=[None]+f_cols, index=(f_cols.index(auto_f_area)+1 if auto_f_area in f_cols else 0), key='f_area')
        if st.button("Run Flood Preprocessing"):
            try:
                f_proc, f_date_used, f_main_used, f_area_used, f_damage_cols = preprocess_df(
                    f_df, date_col=f_date_choice, main_numeric_col=f_main_choice, area_col=f_area_choice)
                st.session_state.flood_df = f_proc
                st.session_state.flood_date = f_date_used
                st.session_state.flood_main = f_main_used
                st.session_state.flood_area = f_area_used
                st.session_state.flood_damage_cols = f_damage_cols
                st.success("Flood preprocessing done")
            except Exception as e:
                st.error(f"Flood preprocessing failed: {e}")

# --------------------
# 3) Comparison & Analysis
# --------------------
elif page == "Comparison & Analysis":
    st.header("3. Comparison & Analysis")
    w = get_df_safe('weather_df')
    f = get_df_safe('flood_df')
    wmain = st.session_state.weather_main
    fmain = st.session_state.flood_main

    st.subheader("Time series")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Weather series")
        st.pyplot(plot_series(w, wmain, highlight_events=True))
    with col2:
        st.write("Flood series")
        st.pyplot(plot_series(f, fmain, highlight_events=True))

# --------------------
# 4) Forecasting
# --------------------
elif page == "Forecasting (Weather)":
    st.header("4. Forecasting - Weather only")
    w = get_df_safe('weather_df')
    wmain = st.session_state.weather_main
    if SARIMAX_AVAILABLE:
        if st.button("Run SARIMA Forecasting"):
            try:
                result = run_sarima(w[wmain])
                st.success(f"SARIMA done - AIC: {result['aic']:.2f}")
                st.write("Forecast plot")
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(result['train'].index, result['train'], label='Train')
                ax.plot(result['test'].index, result['test'], label='Test')
                ax.plot(result['forecast'].index, result['forecast'], label='Forecast')
                ax.set_title(f"SARIMA Forecast - {wmain}")
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Forecasting failed: {e}")
    else:
        st.warning("SARIMAX not installed. Forecasting not available.")

# --------------------
# 5) Summary & Download
# --------------------
elif page == "Summary & Download":
    st.header("5. Summary & Download")
    w = get_df_safe('weather_df')
    f = get_df_safe('flood_df')

    st.subheader("Weather Summary")
    st.write(w.describe())
    st.download_button("Download Weather Processed CSV", data=to_csv_bytes(w), file_name="weather_processed.csv")

    st.subheader("Flood Summary")
    st.write(f.describe())
    st.download_button("Download Flood Processed CSV", data=to_csv_bytes(f), file_name="flood_processed.csv")
