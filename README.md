# Flood Pattern Analysis — Streamlit App

## Overview
This Streamlit app analyzes flood/water-level datasets.  
Features:
- CSV upload (user-provided)
- Column auto-detection + manual override
- Water-level interpolation and flood heuristic
- Visualizations (time series, yearly counts, area impacts)
- Damage aggregation (if present)
- SARIMA forecasting (monthly aggregate)

## Files
- `app.py` — the Streamlit application (single-file)
- `requirements.txt` — Python dependencies
- (Optional) `cleaned_flood_data.csv` — example dataset you can add

## Install & Run Locally
1. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   # Windows
   .venv\\Scripts\\activate
   # macOS / Linux
   source .venv/bin/activate
