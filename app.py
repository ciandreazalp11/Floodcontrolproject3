# app.py
"""
Flood Pattern Analysis — Streamlit single-file app
Features:
- User CSV upload
- Column auto-detection with manual override
- Preprocessing (date parsing, interpolation)
- Flood detection heuristic (z-score + threshold)
- Visualizations (time series, floods per year, avg water per year, top areas)
- Damage analysis (if damage columns exist)
- SARIMA forecasting (requires statsmodels + scikit-learn)
- Save / download summary and output images
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

# Optional SARIMA imports — app will work without them but Forecasting page will warn
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SARIMAX_AVAILABLE = True
except Exception:
    SARIMAX_AVAILABLE = False

# ----------------- Helpers -----------------
@st.cache_data
def load_csv_from_bytes(uploaded_file):
    # Try several encodings
    encodings = ("utf-8", "latin1", "cp1252")
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc, low_memory=False)
        except Exception:
            continue
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, low_memory=False)


def find_col(keywords, cols):
    cols_lower = [c.lower() for c in cols]
    for k in keywords:
        for i, c in enumerate(cols_lower):
            if k in c:
                return cols[i]
    return None


def preprocess_df(df, date_col=None, water_col=None, area_col=None, damage_candidates=None):
    """
    Returns: processed_df, used_date_col, used_water_col, used_area_col, damage_cols
    """
    df = df.copy()
    cols = df.columns.tolist()

    # auto-detect if not supplied
    if date_col is None or date_col == 'None':
        date_col = find_col(['date', 'datetime', 'time', 'day'], cols)
    if water_col is None or water_col == 'None':
        water_col = find_col(['water', 'level', 'wl', 'depth', 'height'], cols)
    if area_col is None or area_col == 'None':
        area_col = find_col(['barangay', 'brgy', 'area', 'location', 'sitio'], cols)

    # Combine Date/Day/Year pattern if present
    if 'Date' in df.columns and 'Day' in df.columns and 'Year' in df.columns:
        df['__combined_date'] = df['Date'].astype(str) + ' ' + df['Day'].astype(str) + ', ' + df['Year'].astype(str)
        date_col = '__combined_date'

    # fallback date
    if date_col is None:
        df['__date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
        date_col = '__date'

    # parse date and set index
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
    df.index = pd.DatetimeIndex(df[date_col])

    # water column fallback: first numeric
    if water_col is None or water_col not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns detected for water level. Please pick a water-level column.")
        water_col = numeric_cols[0]

    # coerce & interpolate water
    df[water_col] = pd.to_numeric(df[water_col], errors='coerce')
    df[water_col] = df[water_col].interpolate(method='linear', limit_direction='both')

    # z-score & flood heuristic
    df['zscore_water'] = stats.zscore(df[water_col].fillna(df[water_col].mean()))
    df['is_outlier_water'] = df['zscore_water'].abs() > 3

    occurrence_col = find_col(['flood', 'event', 'is_flood', 'flooded', 'occurrence'], cols)
    if occurrence_col and occurrence_col in df.columns:
        try:
            df['is_flood'] = df[occurrence_col].astype(bool)
        except Exception:
            df['is_flood'] = df[occurrence_col].notnull()
    else:
        threshold = df[water_col].mean() + 1.0 * df[water_col].std()
        df['is_flood'] = (df[water_col] >= threshold) | (df['zscore_water'].abs() > 1.5)

    df['year'] = df.index.year

    # damage columns
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

    return df, date_col, water_col, area_col, damage_cols


def plot_water_series(df, water_col, highlight_floods=True):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df[water_col], label=str(water_col))
    if highlight_floods and 'is_flood' in df.columns:
        ax.scatter(df.index[df['is_flood']], df[water_col][df['is_flood']], s=20, label='Floods')
    ax.set_title('Water level time series')
    ax.set_xlabel('Date')
    ax.set_ylabel(str(water_col))
    ax.legend()
    plt.tight_layout()
    return fig


def plot_bar_series(series, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4))
    series.plot(kind='bar', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig


def run_sarima_forecast(series, train_frac=0.8, seasonal_period=12):
    if not SARIMAX_AVAILABLE:
        raise RuntimeError('statsmodels SARIMAX or sklearn not available in this environment')

    series_m = series.resample('M').mean().fillna(0)
    if len(series_m) < seasonal_period:
        raise ValueError(f'Not enough aggregated periods for SARIMA (need >= {seasonal_period}).')

    split = int(len(series_m) * float(train_frac))
    train = series_m.iloc[:split]
    test = series_m.iloc[split:]

    best_aic = np.inf
    best_res = None
    best_order = None

    p_range = d_range = q_range = range(0, 2)
    P_range = D_range = Q_range = range(0, 2)

    # small grid for speed
    for p in p_range:
        for d in d_range:
            for q in q_range:
                for P in P_range:
                    for D in D_range:
                        for Q in Q_range:
                            try:
                                mod = SARIMAX(train, order=(p, d, q),
                                              seasonal_order=(P, D, Q, seasonal_period),
                                              enforce_stationarity=False, enforce_invertibility=False)
                                res = mod.fit(disp=False)
                                if res.aic < best_aic:
                                    best_aic = res.aic
                                    best_res = res
                                    best_order = ((p, d, q), (P, D, Q, seasonal_period))
                            except Exception:
                                continue

    if best_res is None:
        raise RuntimeError('SARIMA grid search failed to find a model')

    pred = best_res.get_prediction(start=test.index[0], end=test.index[-1], dynamic=False)
    forecast = pred.predicted_mean
    mae = mean_absolute_error(test, forecast)
    mse = mean_squared_error(test, forecast)

    return {'model': best_res, 'order': best_order, 'aic': best_aic,
            'forecast': forecast, 'train': train, 'test': test, 'mae': mae, 'mse': mse}


# ----------------- Streamlit layout -----------------
st.set_page_config(page_title='Flood Pattern Analysis', layout='wide')
st.title('🌊 Flood Pattern Analysis')

# Sidebar: Upload + options
st.sidebar.header('Data')
uploaded_file = st.sidebar.file_uploader('Upload your flood dataset (CSV)', type=['csv', 'txt'])
use_example = st.sidebar.checkbox('Use example CSV from repo (if available)', value=False)

st.sidebar.header('Options')
show_raw = st.sidebar.checkbox('Show raw dataframe on Preview page', value=False)

st.sidebar.header('Navigation')
page = st.sidebar.radio('Go to:', ['Upload & Preview', 'Preprocessing', 'Analysis & Visualizations', 'Damage Analysis', 'Forecasting (SARIMA)', 'Summary / Download'])

# outputs folder
OUTDIR = 'flood_analysis_outputs'
os.makedirs(OUTDIR, exist_ok=True)

# session variables init
if 'df' not in st.session_state:
    st.session_state.df = None
if 'used_date_col' not in st.session_state:
    st.session_state.used_date_col = None
if 'used_water_col' not in st.session_state:
    st.session_state.used_water_col = None
if 'used_area_col' not in st.session_state:
    st.session_state.used_area_col = None
if 'damage_cols' not in st.session_state:
    st.session_state.damage_cols = []

# ----------------- Pages -----------------
# 1) Upload & Preview
if page == 'Upload & Preview':
    st.header('📤 Upload & Preview Dataset')

    if uploaded_file is not None:
        try:
            df = load_csv_from_bytes(uploaded_file)
            st.session_state.df = df
            st.success('✅ File loaded into session')
            st.write('Shape:', df.shape)
            if show_raw:
                st.dataframe(df.head(100))
        except Exception as e:
            st.error(f'Failed to read uploaded file: {e}')
            st.stop()
    elif use_example:
        example_path = 'cleaned_flood_data.csv'
        if os.path.exists(example_path):
            try:
                df = pd.read_csv(example_path, encoding='latin1', low_memory=False)
                st.session_state.df = df
                st.success(f'Loaded example CSV: {example_path}')
                if show_raw:
                    st.dataframe(df.head(100))
            except Exception as e:
                st.error(f'Failed to read example CSV: {e}')
        else:
            st.info('No example CSV found in the app folder. Upload one via the uploader.')
    else:
        st.info('Upload a CSV using the sidebar file uploader, or enable "Use example CSV" if you added one to the repo.')

# 2) Preprocessing
elif page == 'Preprocessing':
    st.header('⚙️ Preprocessing & Column selection')

    if st.session_state.df is None:
        st.warning('Please upload a CSV first in the "Upload & Preview" page.')
        st.stop()

    df = st.session_state.df.copy()
    cols = df.columns.tolist()

    # auto detection
    detected_date = find_col(['date', 'datetime', 'time', 'day'], cols)
    detected_water = find_col(['water', 'level', 'wl', 'depth', 'height'], cols)
    detected_area = find_col(['barangay', 'brgy', 'area', 'location', 'sitio'], cols)

    st.write('Auto-detected columns (may be None):')
    st.write('Date:', detected_date, ' | Water-level:', detected_water, ' | Area:', detected_area)

    # allow manual override
    date_choice = st.selectbox('Select date column (or None to auto-fallback)', options=[None] + cols, index=(cols.index(detected_date) + 1 if detected_date in cols else 0))
    water_choice = st.selectbox('Select water-level column (or None to pick first numeric)', options=[None] + cols, index=(cols.index(detected_water) + 1 if detected_water in cols else 0))
    area_choice = st.selectbox('Select area column (optional)', options=[None] + cols, index=(cols.index(detected_area) + 1 if detected_area in cols else 0))

    run_prep = st.button('Run Preprocessing')
    if run_prep:
        try:
            processed, used_date_col, used_water_col, used_area_col, damage_cols = preprocess_df(df, date_col=(date_choice if date_choice is not None else None),
                                                                                                   water_col=(water_choice if water_choice is not None else None),
                                                                                                   area_col=(area_choice if area_choice is not None else None))
            st.session_state.df = processed
            st.session_state.used_date_col = used_date_col
            st.session_state.used_water_col = used_water_col
            st.session_state.used_area_col = used_area_col
            st.session_state.damage_cols = damage_cols
            st.success('Preprocessing complete.')
            st.write('Used Date col:', used_date_col)
            st.write('Used Water col:', used_water_col)
            st.write('Used Area col:', used_area_col)
            if show_raw:
                st.dataframe(processed.head(100))
        except Exception as e:
            st.error(f'Preprocessing failed: {e}')

# 3) Analysis & Visualizations
elif page == 'Analysis & Visualizations':
    st.header('📊 Analysis & Visualizations')

    if st.session_state.df is None or st.session_state.used_water_col is None:
        st.warning('Please upload and preprocess data first.')
        st.stop()

    df = st.session_state.df
    used_water_col = st.session_state.used_water_col
    used_area_col = st.session_state.used_area_col

    # Water-level time series
    st.subheader('Water level time series')
    fig_w = plot_water_series(df, used_water_col, highlight_floods=True)
    st.pyplot(fig_w)
    fig_w.savefig(os.path.join(OUTDIR, 'water_level_timeseries.png'))

    # Floods per year
    st.subheader('Flood occurrences per year')
    floods_per_year = df.groupby('year')['is_flood'].sum().astype(int)
    st.bar_chart(floods_per_year)
    # Save image
    fig_fpy = plot_bar_series(floods_per_year, 'Flood occurrences per year', 'Year', 'Number of flood records')
    fig_fpy.savefig(os.path.join(OUTDIR, 'floods_per_year.png'))

    # Avg water per year
    st.subheader('Average water level per year')
    avg_water_per_year = df.groupby('year')[used_water_col].mean()
    st.bar_chart(avg_water_per_year)
    fig_awpy = plot_bar_series(avg_water_per_year, 'Average water level per year', 'Year', f'Average {used_water_col}')
    fig_awpy.savefig(os.path.join(OUTDIR, 'avg_water_per_year.png'))

    # Most affected areas
    if used_area_col and used_area_col in df.columns:
        st.subheader('Most affected areas (top 10)')
        most_affected = df[df['is_flood']].groupby(used_area_col)['is_flood'].sum().sort_values(ascending=False).head(10)
        if not most_affected.empty:
            st.bar_chart(most_affected)
            fig_ma = plot_bar_series(most_affected, 'Top affected areas by flood count', 'Area', 'Flood count')
            fig_ma.savefig(os.path.join(OUTDIR, 'most_affected_areas.png'))
        else:
            st.info('No area-based flood records found.')

# 4) Damage Analysis
elif page == 'Damage Analysis':
    st.header('🏚 Damage Analysis (if available)')

    if st.session_state.df is None:
        st.warning('Please upload and preprocess data first.')
        st.stop()

    df = st.session_state.df
    damage_cols = st.session_state.damage_cols if 'damage_cols' in st.session_state else []
    if damage_cols:
        st.write('Detected damage columns:', damage_cols)
        total_damage_per_year = df.groupby('year')[damage_cols].sum().fillna(0)
        st.dataframe(total_damage_per_year)
        # plot
        fig, ax = plt.subplots(figsize=(10, 4))
        for c in total_damage_per_year.columns:
            ax.plot(total_damage_per_year.index, total_damage_per_year[c], marker='o', label=c)
        ax.set_title('Total damage per year (by damage column)')
        ax.set_xlabel('Year')
        ax.set_ylabel('Damage (dataset units)')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        fig.savefig(os.path.join(OUTDIR, 'total_damage_per_year.png'))
    else:
        st.info('No damage-like columns detected. Name them with keywords like damage, loss, agri, infra to be auto-detected.')

# 5) Forecasting
elif page == 'Forecasting (SARIMA)':
    st.header('📈 SARIMA Forecasting')

    if not SARIMAX_AVAILABLE:
        st.error('SARIMAX or sklearn not installed in this environment. Install statsmodels and scikit-learn to use forecasting.')
        st.info('Local install: pip install statsmodels scikit-learn')
        st.stop()

    if st.session_state.df is None or st.session_state.used_water_col is None:
        st.warning('Please upload and preprocess data first.')
        st.stop()

    df = st.session_state.df
    used_water_col = st.session_state.used_water_col

    st.write('Monthly aggregation (resample M) will be used for SARIMA.')
    train_frac = st.slider('Training fraction (train / all)', min_value=0.5, max_value=0.95, value=0.8, step=0.05)
    run = st.button('Run SARIMA (small grid search)')

    if run:
        try:
            with st.spinner('Running SARIMA grid search — this may take a while...'):
                res = run_sarima_forecast(df[used_water_col], train_frac=train_frac)
            st.success('SARIMA completed')
            st.write('Best order:', res['order'])
            st.write('AIC:', float(res['aic']))
            st.write('MAE:', float(res['mae']), 'MSE:', float(res['mse']))

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(res['train'].index, res['train'], label='Train')
            ax.plot(res['test'].index, res['test'], label='Test')
            ax.plot(res['forecast'].index, res['forecast'], label='Forecast')
            ax.legend()
            ax.set_title('SARIMA: Actual vs Forecast')
            plt.tight_layout()
            st.pyplot(fig)
            fig.savefig(os.path.join(OUTDIR, 'sarima_actual_vs_forecast.png'))
        except Exception as e:
            st.error(f'Forecast failed: {e}')

# 6) Summary / Download
elif page == 'Summary / Download':
    st.header('📥 Summary & Export')

    if st.session_state.df is None or st.session_state.used_water_col is None:
        st.warning('Please upload and preprocess data first.')
        st.stop()

    df = st.session_state.df
    used_water_col = st.session_state.used_water_col

    floods_per_year = df.groupby('year')['is_flood'].sum().astype(int)
    avg_water_per_year = df.groupby('year')[used_water_col].mean()

    summary_df = pd.DataFrame({'year': floods_per_year.index,
                               'floods_per_year': floods_per_year.values,
                               'avg_water_per_year': avg_water_per_year.values})
    st.subheader('Summary per year')
    st.dataframe(summary_df)

    csv_bytes = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download summary CSV', data=csv_bytes, file_name='summary_per_year.csv')

    st.write('Saved output files in app folder (flood_analysis_outputs):')
    st.write(os.listdir(OUTDIR))

    # zip outputs
    import zipfile
    zip_path = os.path.join(OUTDIR, 'flood_outputs.zip')
    with zipfile.ZipFile(zip_path, 'w') as z:
        for fname in os.listdir(OUTDIR):
            z.write(os.path.join(OUTDIR, fname), arcname=fname)
    with open(zip_path, 'rb') as f:
        st.download_button('Download outputs ZIP', data=f.read(), file_name='flood_outputs.zip')

# Footer
st.sidebar.markdown('---')
st.sidebar.write('Tip: if auto-detection picks wrong columns, run Preprocessing and override column choices there.')
