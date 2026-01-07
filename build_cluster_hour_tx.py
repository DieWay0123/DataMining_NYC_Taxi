from __future__ import annotations
from pathlib import Path
import polars as pl

DATASET_DIR = "artifacts/trips_with_cluster_parquet"     # parquet folder
PROFILES_PATH = "artifacts/cluster_profiles.parquet"
OUT_PATH = "artifacts/cluster_hour_tx.parquet"

def payment_type_expr():
    pt = pl.col("payment_type").cast(pl.Int64)
    return (
        pl.when(pt == 1).then(pl.lit("Credit Card"))
        .when(pt == 2).then(pl.lit("Cash"))
        .when(pt == 3).then(pl.lit("No charge"))
        .when(pt == 4).then(pl.lit("Dispute"))
        .when(pt == 5).then(pl.lit("Unknown"))
        .when(pt == 6).then(pl.lit("Voided trip"))
        .otherwise(pl.lit("Unknown"))
        .alias("PaymentType")
    )


def safe_collect(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(streaming=True)
    except TypeError:
        return lf.collect()


def main():
    ds = Path(DATASET_DIR)
    if not ds.exists():
        raise FileNotFoundError(f"Dataset not found: {ds}")

    lf = pl.scan_parquet(str(ds))

    schema = lf.collect_schema()
    if schema.get("tpep_pickup_datetime") == pl.Utf8:
        lf = lf.with_columns(
            pl.col("tpep_pickup_datetime").str.strptime(pl.Datetime, strict=False)
        )

    # 只保留會用到的欄位
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

    # --- 建離散化欄位 + pickup_hour ---
    lf2 = (
        lf.with_columns(
            pl.col("tpep_pickup_datetime").dt.truncate("1h").alias("pickup_hour"),
            pl.col("tpep_pickup_datetime").dt.hour().alias("_h"),
            tip_percent.alias("_tip_pct"),
        )
        .with_columns(
            # Pickup_Time: [0,6,12,18,24] include_lowest=True 的對應
            pl.when(pl.col("_h") <= 6).then(pl.lit("LateNight"))
            .when(pl.col("_h") <= 12).then(pl.lit("Morning"))
            .when(pl.col("_h") <= 18).then(pl.lit("Afternoon"))
            .otherwise(pl.lit("Evening"))
            .alias("Pickup_Time"),

            pl.when(pl.col("_tip_pct") <= med_tip).then(pl.lit("LowTip"))
            .otherwise(pl.lit("HighTip"))
            .alias("Tip_Level"),

            pl.when(pl.col("passenger_count") == 1).then(pl.lit("Solo"))
            .otherwise(pl.lit("Group"))
            .alias("Passenger"),

            pl.when(pl.col("trip_distance") <= med_dist).then(pl.lit("Near"))
            .otherwise(pl.lit("Far"))
            .alias("TripDistance"),

            payment_type_expr(),

            pl.when(pl.col("total_amount") <= med_total).then(pl.lit("LowTotalAmount"))
            .otherwise(pl.lit("HighTotalAmount"))
            .alias("TotalAmount"),

            pl.when(pl.col("fare_amount") <= med_fare).then(pl.lit("LowFareAmount"))
            .otherwise(pl.lit("HighFareAmount"))
            .alias("FareAmount"),
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
        if dtype is not None and dtype.__class__.__name__.lower().startswith("list"):
            out = out.with_columns(pl.col(c).list.first().alias(c))

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_PATH)
    print("saved ->", OUT_PATH, "rows=", out.height)

if __name__ == "__main__":
    main()