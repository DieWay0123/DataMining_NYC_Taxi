import os
import glob
import math
import joblib
import numpy as np
import pandas as pd

# ========= DATA Path =========
RAW_DIR = "data/raw"
KMEANS_PATH = "models/full_partial_fit_k60/kmeans.joblib"
SCALER_PATH = "models/full_partial_fit_k60/scaler.joblib"
OUT_DIR = "artifacts/trips_with_cluster_parquet"
# =================================

# Apriori 需要的欄位 + 時間
WANT_COLS = [
    "tpep_pickup_datetime",
    "pickup_longitude", "pickup_latitude",
    "passenger_count", "trip_distance",
    "payment_type",
    "fare_amount", "tip_amount", "total_amount",
]

LON_MIN, LON_MAX = -75.0, -72.0
LAT_MIN, LAT_MAX = 40.0, 42.0

CHUNKSIZE = 400_000

def pick_usecols(csv_path: str, want_cols: list[str]) -> list[str]:
    header = pd.read_csv(csv_path, nrows=0)
    cols = [c for c in want_cols if c in header.columns]
    missing = set(want_cols) - set(cols)
    if missing:
        print(f"[WARN] {os.path.basename(csv_path)} missing cols: {sorted(missing)}")
    return cols

def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # datetime
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")

    # numeric
    for c in ["pickup_longitude","pickup_latitude","passenger_count","trip_distance",
              "payment_type","fare_amount","tip_amount","total_amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop bad rows
    df = df.dropna(subset=["tpep_pickup_datetime","pickup_longitude","pickup_latitude"])
    df = df[df["pickup_longitude"].between(LON_MIN, LON_MAX) & df["pickup_latitude"].between(LAT_MIN, LAT_MAX)]

    if "total_amount" in df.columns:
        df = df[df["total_amount"].fillna(0) > 0]

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.floor("h")

    return df

def main():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading models...")
    kmeans = joblib.load(KMEANS_PATH)
    scaler = joblib.load(SCALER_PATH)

    csvs = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {RAW_DIR}")

    part_idx = 0
    total_rows = 0

    for csv_path in csvs:
        usecols = pick_usecols(csv_path, WANT_COLS)
        if "pickup_longitude" not in usecols or "pickup_latitude" not in usecols or "tpep_pickup_datetime" not in usecols:
            print(f"[SKIP] {csv_path} (missing lon/lat/datetime)")
            continue

        print(f"\nProcessing: {os.path.basename(csv_path)}  usecols={usecols}")

        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=CHUNKSIZE, low_memory=False):
            chunk = clean_chunk(chunk)
            if chunk.empty:
                continue

            # predict cluster_id
            X = chunk[["pickup_longitude", "pickup_latitude"]].to_numpy(dtype=np.float32)
            Xs = scaler.transform(X)
            chunk["cluster_id"] = kmeans.predict(Xs).astype(np.int16)

            # 只保留後續需要的欄位
            keep = [c for c in [
                "cluster_id", "pickup_hour", "tpep_pickup_datetime",
                "pickup_longitude", "pickup_latitude",
                "passenger_count", "trip_distance", "payment_type",
                "fare_amount", "tip_amount", "total_amount"
            ] if c in chunk.columns]
            chunk = chunk[keep]

            out_path = os.path.join(OUT_DIR, f"part-{part_idx:05d}.parquet")
            chunk.to_parquet(out_path, index=False)
            part_idx += 1
            total_rows += len(chunk)

            print(f"  wrote {os.path.basename(out_path)}  rows={len(chunk):,}")

    print(f"\nDone. wrote parts={part_idx:,}  total_rows={total_rows:,}")
    print(f"Parquet dataset dir: {OUT_DIR}")
    print("You can read it by: pd.read_parquet(OUT_DIR)")

if __name__ == "__main__":
    main()
