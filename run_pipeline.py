from __future__ import annotations

from pathlib import Path
import argparse
import os
import glob

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import polars as pl


# =========================
# Paths / Defaults
# =========================
RAW_DIR = Path("data/raw")
BASE_FEAT_DIR = Path("data/features")
BASE_MODEL_DIR = Path("models")

# NYC bbox
LON_MIN, LON_MAX = -74.3, -73.6
LAT_MIN, LAT_MAX = 40.45, 41.05

# stage1 defaults
K_DEFAULT = 60
CHUNK_DEFAULT = 500_000

# trips_with_cluster defaults
TRIPS_OUT_DIR_DEFAULT = Path("artifacts/trips_with_cluster_parquet")
TRIPS_CHUNK_DEFAULT = 400_000

# cluster_hour_tx output
CLUSTER_HOUR_TX_OUT_DEFAULT = Path("artifacts/cluster_hour_tx.parquet")


# =========================
# run_pipeline.py (stage1 + stage2)
# =========================

# 保留欄位
CANDIDATE_COLS = [
    "tpep_pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "total_amount",
    "fare_amount",
    "tip_amount",
    "trip_distance",
    "passenger_count",
]


def get_dirs(exp: str) -> tuple[Path, Path]:
    feat_dir = BASE_FEAT_DIR / exp
    model_dir = BASE_MODEL_DIR / exp
    feat_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    return feat_dir, model_dir


def list_raw_files() -> list[Path]:
    paths = sorted(RAW_DIR.glob("yellow_tripdata_2016-*.csv"))
    if not paths:
        raise FileNotFoundError("data/raw/ 找不到 yellow_tripdata_2016-*.csv")
    return paths


def iter_csv_chunks(paths: list[Path], chunksize: int = CHUNK_DEFAULT):
    for p in paths:
        header = pd.read_csv(p, nrows=0).columns.tolist()
        use_cols = [c for c in CANDIDATE_COLS if c in header]
        if "tpep_pickup_datetime" not in use_cols:
            raise ValueError(f"{p.name} 缺少 tpep_pickup_datetime，請確認欄位名稱。")
        if not ("pickup_longitude" in use_cols and "pickup_latitude" in use_cols):
            raise ValueError(f"{p.name} 缺少 pickup_longitude/latitude，請確認欄位名稱。")
        for chunk in pd.read_csv(p, usecols=use_cols, chunksize=chunksize):
            yield chunk


def iter_datetime_only(paths: list[Path], chunksize: int):
    for p in paths:
        header = pd.read_csv(p, nrows=0).columns.tolist()
        if "tpep_pickup_datetime" not in header:
            raise ValueError(f"{p.name} 缺少 tpep_pickup_datetime，請確認欄位名稱。")
        for chunk in pd.read_csv(p, usecols=["tpep_pickup_datetime"], chunksize=chunksize):
            yield chunk


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 處理pickup_datetime & clean NaN value
    df["pickup_dt"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_dt"])

    # mask: position bbox filter
    lon = df["pickup_longitude"]
    lat = df["pickup_latitude"]
    mask = lon.between(LON_MIN, LON_MAX) & lat.between(LAT_MIN, LAT_MAX)

    # revenue
    if "total_amount" in df.columns:
        mask &= df["total_amount"] > 0
        df["revenue"] = df["total_amount"]
    elif "fare_amount" in df.columns:
        mask &= df["fare_amount"] > 0
        df["revenue"] = df["fare_amount"]
    else:
        raise ValueError("找不到 total_amount / fare_amount欄位")

    # distance / passenger sanity
    if "trip_distance" in df.columns:
        mask &= df["trip_distance"] > 0
    if "passenger_count" in df.columns:
        mask &= df["passenger_count"].between(1, 6)

    df = df.loc[mask].copy()
    if df.empty:
        return df

    df["pickup_hour"] = df["pickup_dt"].dt.floor("h")
    return df[["pickup_hour", "pickup_longitude", "pickup_latitude", "revenue"]]


def save_models_and_centers(
    scaler: StandardScaler,
    km: MiniBatchKMeans,
    feat_dir: Path,
    model_dir: Path,
):
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(km, model_dir / "kmeans.joblib")

    centers = scaler.inverse_transform(km.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=["lon", "lat"])
    centers_df["cluster_id"] = np.arange(len(centers_df), dtype=int)
    centers_df.to_parquet(feat_dir / "cluster_centers.parquet", index=False)

    print(f"[stage1] saved models -> {model_dir}")
    print(f"[stage1] saved cluster_centers -> {feat_dir / 'cluster_centers.parquet'}")


def models_exist(exp: str) -> bool:
    _, model_dir = get_dirs(exp)
    return (model_dir / "scaler.joblib").exists() and (model_dir / "kmeans.joblib").exists()


def stage1_uniform_by_day(
    *,
    paths: list[Path],
    exp: str,
    k: int,
    sample_target: int,
    chunksize: int,
    batch_size: int,
    seed: int,
):
    feat_dir, model_dir = get_dirs(exp)
    rng = np.random.default_rng(seed)

    all_days = set()
    for chunk in iter_datetime_only(paths, chunksize=chunksize):
        dt = pd.to_datetime(chunk["tpep_pickup_datetime"], errors="coerce").dropna()
        if not dt.empty:
            all_days.update(dt.dt.date.unique().tolist())
    days = sorted(all_days)
    if not days:
        raise ValueError("找不到任何有效日期，請確認 tpep_pickup_datetime 欄位與資料格式。")

    n_days = len(days)
    if sample_target <= 0:
        raise ValueError("--sample-target 必須 > 0")

    base = sample_target // n_days
    rem = sample_target % n_days
    quotas = {d: base for d in days}
    for d in days[:rem]:
        quotas[d] += 1

    reservoirs = {d: np.empty((q, 2), dtype=np.float32) for d, q in quotas.items() if q > 0}
    filled = {d: 0 for d in reservoirs.keys()}
    seen = {d: 0 for d in reservoirs.keys()}

    for chunk in iter_csv_chunks(paths, chunksize=chunksize):
        c = clean_chunk(chunk)
        if c.empty:
            continue

        day_series = c["pickup_hour"].dt.date
        coords = c[["pickup_longitude", "pickup_latitude"]].to_numpy(dtype=np.float32)

        for i in range(len(c)):
            d = day_series.iat[i]
            if d not in reservoirs:
                continue

            seen[d] += 1
            q = reservoirs[d].shape[0]

            if filled[d] < q:
                reservoirs[d][filled[d]] = coords[i]
                filled[d] += 1
            else:
                j = rng.integers(0, seen[d])
                if j < q:
                    reservoirs[d][j] = coords[i]

    blocks = []
    for d in reservoirs.keys():
        if filled[d] > 0:
            blocks.append(reservoirs[d][: filled[d]])
    if not blocks:
        raise ValueError("抽樣結果為空")

    coords_sample = np.vstack(blocks)
    print(f"[stage1 uniform_by_day] sampled {coords_sample.shape[0]:,} points over {n_days} days")

    scaler = StandardScaler()
    X = scaler.fit_transform(coords_sample)

    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        random_state=seed,
        n_init="auto",
    )
    km.fit(X)

    save_models_and_centers(scaler, km, feat_dir, model_dir)


def stage1_full_partial_fit(
    *,
    paths: list[Path],
    exp: str,
    k: int,
    chunksize: int,
    batch_size: int,
    seed: int,
):
    feat_dir, model_dir = get_dirs(exp)

    scaler = StandardScaler()
    total_seen = 0
    for chunk in iter_csv_chunks(paths, chunksize=chunksize):
        c = clean_chunk(chunk)
        if c.empty:
            continue
        coords = c[["pickup_longitude", "pickup_latitude"]].to_numpy(dtype=np.float32)
        scaler.partial_fit(coords)
        total_seen += len(coords)
    print(f"[stage1 full_partial_fit] scaler.partial_fit done (seen={total_seen:,})")

    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        random_state=seed,
        n_init="auto",
    )

    total_seen2 = 0
    for chunk in iter_csv_chunks(paths, chunksize=chunksize):
        c = clean_chunk(chunk)
        if c.empty:
            continue
        coords = c[["pickup_longitude", "pickup_latitude"]].to_numpy(dtype=np.float32)
        X = scaler.transform(coords)
        km.partial_fit(X)
        total_seen2 += len(coords)
    print(f"[stage1 full_partial_fit] kmeans.partial_fit done (seen={total_seen2:,})")

    save_models_and_centers(scaler, km, feat_dir, model_dir)


def stage2_aggregate(
    *,
    paths: list[Path],
    exp: str,
    chunksize: int,
):
    feat_dir, model_dir = get_dirs(exp)

    scaler: StandardScaler = joblib.load(model_dir / "scaler.joblib")
    km: MiniBatchKMeans = joblib.load(model_dir / "kmeans.joblib")

    agg: dict[tuple[pd.Timestamp, int], list[float]] = {}

    for chunk in iter_csv_chunks(paths, chunksize=chunksize):
        c = clean_chunk(chunk)
        if c.empty:
            continue

        coords = c[["pickup_longitude", "pickup_latitude"]].to_numpy(dtype=np.float32)
        X = scaler.transform(coords)
        cluster_id = km.predict(X)
        c["cluster_id"] = cluster_id.astype(int)

        g = c.groupby(["pickup_hour", "cluster_id"], as_index=False).agg(
            demand=("cluster_id", "size"),
            revenue=("revenue", "sum"),
        )

        for row in g.itertuples(index=False):
            key = (row.pickup_hour, int(row.cluster_id))
            if key not in agg:
                agg[key] = [int(row.demand), float(row.revenue)]
            else:
                agg[key][0] += int(row.demand)
                agg[key][1] += float(row.revenue)

    out = pd.DataFrame(
        [(k[0], k[1], v[0], v[1]) for k, v in agg.items()],
        columns=["pickup_hour", "cluster_id", "demand", "revenue"],
    ).sort_values(["pickup_hour", "cluster_id"])

    out.to_parquet(feat_dir / "zone_hour.parquet", index=False)
    print(f"[stage2] saved zone_hour -> {feat_dir / 'zone_hour.parquet'}")


# =========================
# build_trips_with_cluster.py
# =========================
TRIPS_WANT_COLS = [
    "tpep_pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "passenger_count",
    "trip_distance",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "total_amount",
]


def pick_usecols_tx(csv_path: str, want_cols: list[str]) -> list[str]:
    header = pd.read_csv(csv_path, nrows=0)
    cols = [c for c in want_cols if c in header.columns]
    missing = set(want_cols) - set(cols)
    if missing:
        print(f"[WARN] {os.path.basename(csv_path)} missing cols: {sorted(missing)}")
    return cols


def clean_chunk_tx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")

    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "passenger_count",
        "trip_distance",
        "payment_type",
        "fare_amount",
        "tip_amount",
        "total_amount",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["tpep_pickup_datetime", "pickup_longitude", "pickup_latitude"])
    df = df[df["pickup_longitude"].between(LON_MIN, LON_MAX) & df["pickup_latitude"].between(LAT_MIN, LAT_MAX)]

    if "total_amount" in df.columns:
        df = df[df["total_amount"].fillna(0) > 0]

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.floor("h")
    return df


def build_trips_with_cluster(
    *,
    paths: list[Path],
    model_dir: Path,
    out_dir: Path,
    chunksize: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    kmeans = joblib.load(model_dir / "kmeans.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")

    part_idx = 0
    total_rows = 0

    for p in paths:
        csv_path = str(p)
        usecols = pick_usecols_tx(csv_path, TRIPS_WANT_COLS)

        if "pickup_longitude" not in usecols or "pickup_latitude" not in usecols or "tpep_pickup_datetime" not in usecols:
            print(f"[SKIP] {p.name} (missing lon/lat/datetime)")
            continue

        print(f"\n[trips_with_cluster] Processing: {p.name}  usecols={usecols}")

        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
            chunk = clean_chunk_tx(chunk)
            if chunk.empty:
                continue

            X = chunk[["pickup_longitude", "pickup_latitude"]].to_numpy(dtype=np.float32)
            Xs = scaler.transform(X)
            chunk["cluster_id"] = kmeans.predict(Xs).astype(np.int16)

            keep = [
                c
                for c in [
                    "cluster_id",
                    "pickup_hour",
                    "tpep_pickup_datetime",
                    "pickup_longitude",
                    "pickup_latitude",
                    "passenger_count",
                    "trip_distance",
                    "payment_type",
                    "fare_amount",
                    "tip_amount",
                    "total_amount",
                ]
                if c in chunk.columns
            ]
            chunk = chunk[keep]

            out_path = out_dir / f"part-{part_idx:05d}.parquet"
            chunk.to_parquet(out_path, index=False)
            part_idx += 1
            total_rows += len(chunk)

            print(f"  wrote {out_path.name}  rows={len(chunk):,}")

    print(f"\n[trips_with_cluster] Done. wrote parts={part_idx:,}  total_rows={total_rows:,}")
    print(f"[trips_with_cluster] Parquet dataset dir: {out_dir}")


# =========================
# build_cluster_hour_tx.py
# =========================
def payment_type_expr():
    pt = pl.col("payment_type").cast(pl.Int64)
    return (
        pl.when(pt == 1)
        .then(pl.lit("Credit Card"))
        .when(pt == 2)
        .then(pl.lit("Cash"))
        .when(pt == 3)
        .then(pl.lit("No charge"))
        .when(pt == 4)
        .then(pl.lit("Dispute"))
        .when(pt == 5)
        .then(pl.lit("Unknown"))
        .when(pt == 6)
        .then(pl.lit("Voided trip"))
        .otherwise(pl.lit("Unknown"))
        .alias("PaymentType")
    )


def safe_collect(lf: "pl.LazyFrame") -> "pl.DataFrame":
    try:
        return lf.collect(streaming=True)
    except TypeError:
        return lf.collect()


def build_cluster_hour_tx(*, dataset_dir: Path, out_path: Path):
    if pl is None:
        raise RuntimeError("polars 未安裝：cluster_hour_tx 需要 polars。請先 uv add polars")

    ds = Path(dataset_dir)
    if not ds.exists():
        raise FileNotFoundError(f"Dataset not found: {ds}")

    lf = pl.scan_parquet(str(ds))

    schema = lf.collect_schema()
    if schema.get("tpep_pickup_datetime") == pl.Utf8:
        lf = lf.with_columns(pl.col("tpep_pickup_datetime").str.strptime(pl.Datetime, strict=False))

    need_cols = [
        "cluster_id",
        "tpep_pickup_datetime",
        "passenger_count",
        "trip_distance",
        "payment_type",
        "fare_amount",
        "tip_amount",
        "total_amount",
    ]
    keep = [c for c in need_cols if schema.get(c) is not None]
    lf = lf.select(keep)

    tip_percent = (
        pl.when(pl.col("total_amount") > 0)
        .then(pl.col("tip_amount") / pl.col("total_amount") * 100.0)
        .otherwise(0.0)
        .fill_null(0.0)
    )

    stats = safe_collect(
        lf.select(
            tip_percent.median().alias("med_tip_pct"),
            pl.col("trip_distance").median().alias("med_dist"),
            pl.col("total_amount").median().alias("med_total"),
            pl.col("fare_amount").median().alias("med_fare"),
        )
    )

    med_tip = float(stats["med_tip_pct"][0])
    med_dist = float(stats["med_dist"][0])
    med_total = float(stats["med_total"][0])
    med_fare = float(stats["med_fare"][0])

    lf2 = (
        lf.with_columns(
            pl.col("tpep_pickup_datetime").dt.truncate("1h").alias("pickup_hour"),
            pl.col("tpep_pickup_datetime").dt.hour().alias("_h"),
            tip_percent.alias("_tip_pct"),
        )
        .with_columns(
            pl.when(pl.col("_h") <= 6)
            .then(pl.lit("LateNight"))
            .when(pl.col("_h") <= 12)
            .then(pl.lit("Morning"))
            .when(pl.col("_h") <= 18)
            .then(pl.lit("Afternoon"))
            .otherwise(pl.lit("Evening"))
            .alias("Pickup_Time"),
            pl.when(pl.col("_tip_pct") <= med_tip).then(pl.lit("LowTip")).otherwise(pl.lit("HighTip")).alias("Tip_Level"),
            pl.when(pl.col("passenger_count") == 1).then(pl.lit("Solo")).otherwise(pl.lit("Group")).alias("Passenger"),
            pl.when(pl.col("trip_distance") <= med_dist).then(pl.lit("Near")).otherwise(pl.lit("Far")).alias("TripDistance"),
            payment_type_expr(),
            pl.when(pl.col("total_amount") <= med_total)
            .then(pl.lit("LowTotalAmount"))
            .otherwise(pl.lit("HighTotalAmount"))
            .alias("TotalAmount"),
            pl.when(pl.col("fare_amount") <= med_fare).then(pl.lit("LowFareAmount")).otherwise(pl.lit("HighFareAmount")).alias("FareAmount"),
        )
        .select(
            "cluster_id",
            "pickup_hour",
            "Pickup_Time",
            "Tip_Level",
            "Passenger",
            "TripDistance",
            "PaymentType",
            "TotalAmount",
            "FareAmount",
        )
    )

    agg = lf2.group_by(["cluster_id", "pickup_hour"]).agg(
        pl.len().alias("trips"),
        pl.col("Pickup_Time").mode().alias("Pickup_Time"),
        pl.col("Tip_Level").mode().alias("Tip_Level"),
        pl.col("Passenger").mode().alias("Passenger"),
        pl.col("TripDistance").mode().alias("TripDistance"),
        pl.col("PaymentType").mode().alias("PaymentType"),
        pl.col("TotalAmount").mode().alias("TotalAmount"),
        pl.col("FareAmount").mode().alias("FareAmount"),
    )

    out = safe_collect(agg)

    cat_cols = ["Pickup_Time", "Tip_Level", "Passenger", "TripDistance", "PaymentType", "TotalAmount", "FareAmount"]
    for c in cat_cols:
        dtype = out.schema.get(c)
        if dtype is not None and str(dtype).startswith("List"):
            out = out.with_columns(pl.col(c).list.first().alias(c))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    print("saved ->", out_path, "rows=", out.height)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    # ===== run_pipeline_a args =====
    parser.add_argument("--exp", type=str, required=True, help="輸出資料夾名稱")
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--chunksize", type=int, default=CHUNK_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refit", action="store_true", help="強制重新訓練(覆蓋該exp路徑下模型)")

    parser.add_argument(
        "--clustering-mode",
        choices=["uniform_by_day", "full_partial_fit"],
        default="uniform_by_day",
        help="stage1 clustering 方式",
    )
    parser.add_argument(
        "--sample-target",
        type=int,
        default=800_000,
        help="uniform_by_day 的總抽樣點數(full_partial_fit 不使用)",
    )

    # ===== integrated steps args =====
    parser.add_argument(
        "--skip-trips-with-cluster",
        action="store_true",
        help="不產生 artifacts/trips_with_cluster_parquet dataset(Apriori 用)",
    )
    parser.add_argument(
        "--trips-chunksize",
        type=int,
        default=TRIPS_CHUNK_DEFAULT,
        help="build_trips_with_cluster 的 chunksize",
    )
    parser.add_argument(
        "--trips-out-dir",
        type=str,
        default=str(TRIPS_OUT_DIR_DEFAULT),
        help="build_trips_with_cluster 的輸出資料夾(parquet dataset)",
    )

    parser.add_argument(
        "--skip-cluster-hour-tx",
        action="store_true",
        help="不產生 cluster_hour_tx.parquet(Apriori 用的交易表)",
    )
    parser.add_argument(
        "--cluster-hour-tx-out",
        type=str,
        default=str(CLUSTER_HOUR_TX_OUT_DEFAULT),
        help="cluster_hour_tx.parquet 輸出位置",
    )

    args = parser.parse_args()

    paths = list_raw_files()
    print("raw files:", [p.name for p in paths])

    # ===== stage1 =====
    if args.refit or (not models_exist(args.exp)):
        if args.clustering_mode == "uniform_by_day":
            stage1_uniform_by_day(
                paths=paths,
                exp=args.exp,
                k=args.k,
                sample_target=args.sample_target,
                chunksize=args.chunksize,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        else:
            stage1_full_partial_fit(
                paths=paths,
                exp=args.exp,
                k=args.k,
                chunksize=args.chunksize,
                batch_size=args.batch_size,
                seed=args.seed,
            )
    else:
        print(f"[stage1] models already exist for exp='{args.exp}'. (use --refit to retrain)")

    # ===== stage2 aggregate (dashboard 用) =====
    stage2_aggregate(paths=paths, exp=args.exp, chunksize=args.chunksize)

    # ===== stage3 trips_with_cluster dataset (Apriori 用) =====
    feat_dir, model_dir = get_dirs(args.exp)
    trips_out_dir = Path(args.trips_out_dir)

    if not args.skip_trips_with_cluster:
        build_trips_with_cluster(
            paths=paths,
            model_dir=model_dir,
            out_dir=trips_out_dir,
            chunksize=args.trips_chunksize,
        )

    # ===== stage4 cluster_hour_tx.parquet (Apriori 交易表) =====
    if not args.skip_cluster_hour_tx:
        build_cluster_hour_tx(
            dataset_dir=trips_out_dir,
            out_path=Path(args.cluster_hour_tx_out),
        )


if __name__ == "__main__":
    main()
