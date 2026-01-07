from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import joblib

RAW_DIR = Path("data/raw")
BASE_FEAT_DIR = Path("data/features")
BASE_MODEL_DIR = Path("models")

# bbox
LON_MIN, LON_MAX = -74.3, -73.6
LAT_MIN, LAT_MAX = 40.45, 41.05

# for k-means
K = 60
CHUNK = 500_000

# 保留欄位
CANDIDATE_COLS = [
    "tpep_pickup_datetime",
    "pickup_longitude", "pickup_latitude",
    "total_amount", "fare_amount", "tip_amount",
    "trip_distance", "passenger_count",
]

def get_dirs(exp: str) -> tuple[Path, Path]:
    """Return (feat_dir, model_dir) for the experiment name."""
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

# 讀csv
def iter_csv_chunks(paths: list[Path], chunksize: int = CHUNK):
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

# 資料清理 + 特徵處理
def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 處理pickup_datetime & clean NaN value    
    df["pickup_dt"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_dt"])

    # mask: position bbox filter
    lon = df["pickup_longitude"]
    lat = df["pickup_latitude"]
    mask = lon.between(LON_MIN, LON_MAX) & lat.between(LAT_MIN, LAT_MAX)

    # 處理每趟計程車的營業額資訊(若total_amount資料錯誤則取fare_amount)
    if "total_amount" in df.columns:
        mask &= df["total_amount"] > 0
        df["revenue"] = df["total_amount"]
    elif "fare_amount" in df.columns:
        mask &= df["fare_amount"] > 0
        df["revenue"] = df["fare_amount"]
    else:
        raise ValueError("找不到 total_amount / fare_amount欄位")

    # 處理乘車距離&乘車人數欄位
    if "trip_distance" in df.columns:
        mask &= df["trip_distance"] > 0
    if "passenger_count" in df.columns:
        mask &= df["passenger_count"].between(1, 6) # 刪去不合理資料(一台計程車差不多最多坐6人)

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

# =========================
# Clustering Pipelines
# =========================

# 每個chunk取一定數量資料的clustering，降低尖峰時段影響
def stage1_uniform_by_day(
    *,
    paths: list[Path],
    exp: str,
    k: int,
    sample_target: int, # 每個chunk要取多少資料
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
    
    # ============================================
    
    # 分配每一天選取的資料數量
    # base: 每一天基礎選取數量
    # rem: 剩下沒辦法整除平均分配的部分讓所有天的前rem天quota+1
    base = sample_target // n_days
    rem = sample_target % n_days
    quotas = {d: base for d in days}
    for d in days[:rem]:
        quotas[d] += 1
    
    # q: 每個chunk需要抽樣的資料數
    reservoirs = {d: np.empty((q, 2), dtype=np.float32) for d, q in quotas.items() if q > 0} # 存選中抽樣的座標
    filled = {d: 0 for d in reservoirs.keys()} # 存該date已經選取幾筆抽樣
    seen = {d: 0 for d in reservoirs.keys()} # 存該date已經看過多少趟行程
    
    for chunk in iter_csv_chunks(paths, chunksize=chunksize):
        c = clean_chunk(chunk)
        if c.empty:
            continue

        day_series = c["pickup_hour"].dt.date
        coords = c[["pickup_longitude", "pickup_latitude"]].to_numpy(dtype=np.float32)

        # chunk內開始進行抽樣
        for i in range(len(c)):
            d = day_series.iat[i]
            if d not in reservoirs:
                continue

            seen[d] += 1
            q = reservoirs[d].shape[0]

            # 用抽樣算法確保每個該日行程被選中的機率是相同的, 確保均勻抽樣
            if filled[d] < q:
                reservoirs[d][filled[d]] = coords[i]
                filled[d] += 1
            else:
                # 若抽樣已選取滿, 確保後續資料被選取的機率也是相同的替換機制
                # reservoir replacement
                j = rng.integers(0, seen[d])
                if j < q:
                    reservoirs[d][j] = coords[i]

    # stack all samples
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
    # ============================================

# 全部資料一起做clustering
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

# 將分群規則套用到所有計程車行程資料中
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

        # 計算各chunk資料的demand(需求)和revenue(營收)
        g = c.groupby(["pickup_hour", "cluster_id"], as_index=False).agg(
            demand=("cluster_id", "size"),
            revenue=("revenue", "sum"),
        )

        # 各chunk資料總聚合
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="輸出資料夾名稱")
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--chunksize", type=int, default=500_000)
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
    args = parser.parse_args()
    
    paths = list_raw_files()
    print("raw files:", [p.name for p in paths])
    
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
    
    stage2_aggregate(
        paths=paths,
        exp=args.exp,
        chunksize=args.chunksize,
    )

    
if __name__ == "__main__":
    main()