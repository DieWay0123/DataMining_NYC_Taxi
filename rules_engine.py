from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

PAYMENT_MAP = {
    1: "Credit Card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided trip",
}

DISCRETE_LABELS = {
    "Pickup_Time": ["LateNight", "Morning", "Afternoon", "Evening"],  # :contentReference[oaicite:2]{index=2}
    "Tip_Level": ["LowTip", "HighTip"],                               # :contentReference[oaicite:3]{index=3}
    "Passenger": ["Solo", "Group"],                                   # :contentReference[oaicite:4]{index=4}
    "TripDistance": ["Near", "Far"],                                  # :contentReference[oaicite:5]{index=5}
    "TotalAmount": ["LowTotalAmount", "HighTotalAmount"],             # :contentReference[oaicite:6]{index=6}
    "FareAmount": ["LowFareAmount", "HighFareAmount"],                # :contentReference[oaicite:7]{index=7}
}

def _mode_or_unknown(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    if s.empty:
        return "Unknown"
    return s.value_counts().idxmax()

def discretize_trips_like_teammate(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    if "tpep_pickup_datetime" in df.columns:
        t = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
        hour = t.dt.hour
        out["Pickup_Time"] = pd.cut(
            hour, bins=[0, 6, 12, 18, 24],
            labels=["LateNight", "Morning", "Afternoon", "Evening"],
            include_lowest=True
        )

    if "tip_amount" in df.columns and "total_amount" in df.columns:
        tip = pd.to_numeric(df["tip_amount"], errors="coerce")
        total = pd.to_numeric(df["total_amount"], errors="coerce")
        tip_percent = (tip / total) * 100
        tip_percent = tip_percent.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        med = float(tip_percent.median())
        mx = float(tip_percent.max()) if float(tip_percent.max()) > 0 else 1.0

        out["Tip_Level"] = pd.cut(
            tip_percent, bins=[0.0, med, mx],
            labels=["LowTip", "HighTip"],
            include_lowest=True
        )

    if "passenger_count" in df.columns:
        pc = pd.to_numeric(df["passenger_count"], errors="coerce").fillna(1)
        out["Passenger"] = pc.apply(lambda x: "Solo" if x == 1 else "Group")

    if "trip_distance" in df.columns:
        dist = pd.to_numeric(df["trip_distance"], errors="coerce").fillna(0.0)
        med = float(dist.median())
        mx = float(dist.max()) if float(dist.max()) > 0 else 1.0
        out["TripDistance"] = pd.cut(
            dist, bins=[0.0, med, mx],
            labels=["Near", "Far"],
            include_lowest=True
        )

    if "payment_type" in df.columns:
        pt = pd.to_numeric(df["payment_type"], errors="coerce")
        out["PaymentType"] = pt.map(PAYMENT_MAP).fillna("Unknown")

    if "total_amount" in df.columns:
        total = pd.to_numeric(df["total_amount"], errors="coerce").fillna(0.0)
        med = float(total.median())
        mx = float(total.max()) if float(total.max()) > 0 else 1.0
        out["TotalAmount"] = pd.cut(
            total, bins=[0.0, med, mx],
            labels=["LowTotalAmount", "HighTotalAmount"],
            include_lowest=False
        )

    if "fare_amount" in df.columns:
        fare = pd.to_numeric(df["fare_amount"], errors="coerce").fillna(0.0)
        med = float(fare.median())
        mx = float(fare.max()) if float(fare.max()) > 0 else 1.0
        out["FareAmount"] = pd.cut(
            fare, bins=[0.0, med, mx],
            labels=["LowFareAmount", "HighFareAmount"],
            include_lowest=False
        )

    return out


def build_cluster_hour_transactions(
    trips: pd.DataFrame,
    cluster_col: str = "cluster_id",
    pickup_dt_col: str = "tpep_pickup_datetime",
) -> pd.DataFrame:
    df = trips.copy()
    df[pickup_dt_col] = pd.to_datetime(df[pickup_dt_col], errors="coerce")
    df["pickup_hour"] = df[pickup_dt_col].dt.floor("h")

    # trip-level 離散化
    disc = discretize_trips_like_teammate(df)

    tmp = pd.concat([df[[cluster_col, "pickup_hour"]], disc], axis=1)
    tmp = tmp.dropna(subset=[cluster_col, "pickup_hour"])

    # 聚合到 cluster×hour
    agg_dict = {c: (c, _mode_or_unknown) for c in disc.columns}
    out = tmp.groupby([cluster_col, "pickup_hour"], as_index=False).agg(**agg_dict)
    out["trips"] = tmp.groupby([cluster_col, "pickup_hour"]).size().values

    # 組成 items
    def row_items(r):
        items = []
        items.append(f"CLUSTER={int(r[cluster_col])}")
        if "zone_type" in out.columns:
            items.append(f"ZONE={r['zone_type']}")
        for c in disc.columns:
            items.append(f"{c}={r[c]}")
        return items

    out["items"] = out.apply(row_items, axis=1)
    return out


@dataclass
class AprioriParams:
    min_support: float = 0.05
    metric: str = "lift"
    min_threshold: float = 1.1
    head_n: int = 20
    max_len: int = 3
    sort_by: str = "confidence"


def mine_rules_from_transactions(tx_items: list[list[str]], p: AprioriParams) -> pd.DataFrame:
    te = TransactionEncoder()
    te_ary = te.fit(tx_items).transform(tx_items)
    onehot = pd.DataFrame(te_ary, columns=te.columns_)

    fi = apriori(onehot, min_support=p.min_support, use_colnames=True, max_len=p.max_len)
    rules = association_rules(fi, metric=p.metric, min_threshold=p.min_threshold)

    sort_col = p.sort_by if p.sort_by in rules.columns else p.metric
    rules = rules.sort_values(sort_col, ascending=False)

    result = rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(p.head_n).copy()
    result["antecedents"] = result["antecedents"].apply(lambda x: ", ".join(list(x)))
    result["consequents"] = result["consequents"].apply(lambda x: ", ".join(list(x)))
    return result
