# NYC Yellow Taxi Mining Dashboard (2016 Q1)

A data mining + visualization project using NYC Yellow Taxi trip data (2016/01-03).
Includes:

- Hotspot clustering (KMeans)
- Demand & revenue dashboard (Streamlit)
- Dispatch recommendation (baseline forecast)
- Apriori association rules on cluster×hour transactions

## Tech Stack

- Python + uv
- pandas / scikit-learn
- streamlit + pydeck + altair
- polars (for building cluster×hour transaction table)
- joblib

## Project Structure

- `run_pipeline.py`: build models + features + artifacts
- `app.py`: Streamlit dashboard
- `dispatch.py`: dispatch tab
- `apriori_tab.py`: Apriori insights tab

## Setup (uv)

```bash
uv sync
```

## Data

Download NYC Yellow Taxi Trip Data (e.g., 2016-01 ~ 2016-03) from Kaggle / official sources.

Place raw CSVs here:

```bash
data/raw/yellow_tripdata_2016-01.csv
data/raw/yellow_tripdata_2016-02.csv
data/raw/yellow_tripdata_2016-03.csv
```

## Build Pipeline

```bash
uv run python run run_pipeline.py --exp exp_k60 --clustering-mode full_partial_fit --refit
```

## Run Dashboard

```bash
uv run streamlit run app.py
```
