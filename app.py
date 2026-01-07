from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt
from dispatch import render_dispatch_tab
from apriori_tab import render_apriori_tab


BASE_FEAT_DIR = Path("data/features")
ARTIFACTS_DIR = Path("artifacts")
APR_TX_PATH = ARTIFACTS_DIR / "cluster_hour_tx.parquet"
DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# -------------------------
# Data loading & utilities
# -------------------------
def list_experiments(base: Path) -> list[str]:
    if not base.exists():
        return []
    exps = []
    for p in sorted(base.iterdir()):
        if p.is_dir():
            if (p / "zone_hour.parquet").exists() and (p / "cluster_centers.parquet").exists():
                exps.append(p.name)
    return exps


@st.cache_data
def load_data(exp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    zone = pd.read_parquet(BASE_FEAT_DIR / exp / "zone_hour.parquet")
    centers = pd.read_parquet(BASE_FEAT_DIR / exp / "cluster_centers.parquet")
    zone["pickup_hour"] = pd.to_datetime(zone["pickup_hour"])
    return zone, centers

# 用於計算
def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")

def _softmax(scores: dict[str, float], temperature: float = 0.45) -> dict[str, float]:
    keys = list(scores.keys())
    x = np.array([scores[k] for k in keys], dtype=float) / max(1e-6, float(temperature))
    x = x - np.max(x)
    ex = np.exp(x)
    p = ex / max(1e-12, float(ex.sum()))
    return {k: float(v) for k, v in zip(keys, p)}


def normalize_date_range(date_range):
    # st.date_input 有時可能回傳單一 date
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        return date_range[0], date_range[1]
    return date_range, date_range


def date_filter(zone: pd.DataFrame, date_range) -> pd.DataFrame:
    if date_range is None:
        return zone.iloc[0:0].copy(), False
    
    if not isinstance(date_range, (tuple, list)):
        start_date = date_range
        end_date = None
    else:
        if len(date_range) != 2:
            return zone.iloc[0:0].copy(), False
        start_date, end_date = date_range

    if start_date is None or end_date is None:
        return zone.iloc[0:0].copy(), False
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # include end date

    out = zone[(zone["pickup_hour"] >= start) & (zone["pickup_hour"] < end)].copy()
    return out, True


def cluster_summary(filtered: pd.DataFrame) -> pd.DataFrame:
    byc = filtered.groupby("cluster_id", as_index=False).agg(
        demand=("demand", "sum"),
        revenue=("revenue", "sum"),
    )
    byc["cluster_id"] = byc["cluster_id"].astype(int)
    byc["avg_rev_per_trip"] = byc["revenue"] / byc["demand"].replace(0, np.nan)
    return byc


def safe_log_radius(values: pd.Series, min_r: float, max_r: float) -> np.ndarray:
    v = values.astype(float).clip(lower=0.0)
    z = np.log1p(v)
    zmin, zmax = float(z.min()), float(z.max())
    if abs(zmax - zmin) < 1e-12:
        return np.full(len(v), (min_r + max_r) / 2.0, dtype=float)
    scaled = (z - zmin) / (zmax - zmin)
    return (scaled * (max_r - min_r) + min_r).to_numpy(dtype=float)


def make_map_df(summary: pd.DataFrame, centers: pd.DataFrame, metric: str) -> pd.DataFrame:
    m = summary.merge(centers, on="cluster_id", how="left").dropna(subset=["lon", "lat"])
    m = m.sort_values(metric, ascending=False).reset_index(drop=True)
    m["rank"] = np.arange(1, len(m) + 1)
    m["rank_text"] = m["rank"].astype(str)

    # for chart string
    m["demand_str"] = m["demand"].map(lambda x: f"{int(x):,}")
    m["revenue_str"] = m["revenue"].map(lambda x: f"${float(x):,.0f}")
    m["avg_str"] = m["avg_rev_per_trip"].map(lambda x: "-" if pd.isna(x) else f"${float(x):.2f}")

    # label: use itertuples to avoid pandas Series .rank() method collision
    def make_label(row):
        avg = "-" if pd.isna(row.avg_rev_per_trip) else f"${float(row.avg_rev_per_trip):.2f}"
        return (
            f"#{int(row.rank):02d}  cluster {int(row.cluster_id)}  |  "
            f"demand={int(row.demand):,}  |  revenue=${float(row.revenue):,.0f}  |  avg={avg}"
        )

    m["label"] = [make_label(r) for r in m.itertuples(index=False)]
    return m


def build_scatter_layers(
    map_show: pd.DataFrame,
    selected_cluster: int,
    metric: str,
    bubble_min: int,
    bubble_max: int,
    show_rank_labels: bool,
) -> list[pdk.Layer]:
    df = map_show.copy()
    
    if "zone_type" not in df.columns:
        df["zone_type"] = "Balanced/Uncertain"
    
    df["radius"] = safe_log_radius(df[metric], bubble_min, bubble_max)

    def make_fill(row):
        r, g, b = color_for_zone_type(row["zone_type"])
        is_sel = int(row["cluster_id"]) == int(selected_cluster)
        alpha = 230 if is_sel else 160

        # 選中的cluster凸顯顏色
        if is_sel:
            r, g, b = _clamp255(r * 1.10), _clamp255(g * 1.10), _clamp255(b * 1.10)

        return [r, g, b, alpha]
    df["fill_color"] = df.apply(make_fill, axis=1)
    df["line_color"] = df["cluster_id"].apply(
        lambda cid: [255, 255, 255, 240] if int(cid) == int(selected_cluster) else [0, 0, 0, 0]
    )

    scatter = pdk.Layer(
        "ScatterplotLayer",
        id="hotspots_scatter",
        data=df,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="fill_color",
        get_line_color="line_color",
        stroked=True,
        line_width_min_pixels=2,
        pickable=True,
        auto_highlight=True,
    )

    layers = [scatter]

    if show_rank_labels:
        text = pdk.Layer(
            "TextLayer",
            data=df,
            id="hotspots_text",
            get_position="[lon, lat]",
            get_text="rank_text",
            get_size=16,
            get_color=[20, 20, 20, 230],
            get_text_anchor="'middle'",
            get_alignment_baseline="'center'",
            pickable=False,
        )
        layers.append(text)

    return layers


def build_dow_profile(ts_hourly: pd.DataFrame) -> pd.DataFrame:
    tmp = ts_hourly.copy()
    tmp["dow"] = tmp.index.dayofweek
    prof = tmp.groupby("dow")[["demand", "revenue"]].mean()
    prof = prof.reindex(range(7), fill_value=0.0)
    prof["avg_rev_per_trip"] = prof["revenue"] / prof["demand"].replace(0, np.nan)
    prof.index = DOW_ORDER
    return prof


def build_hod_profile(ts_hourly: pd.DataFrame) -> pd.DataFrame:
    tmp = ts_hourly.copy()
    tmp["hour"] = tmp.index.hour
    prof = tmp.groupby("hour")[["demand", "revenue"]].mean()
    prof = prof.reindex(range(24), fill_value=0.0)
    prof["avg_rev_per_trip"] = prof["revenue"] / prof["demand"].replace(0, np.nan)
    return prof


def altair_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, sort=None):
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:N", sort=sort, title=None),
            y=alt.Y(f"{y_col}:Q", title=None),
            tooltip=list(df.columns),
        )
        .properties(title=title)
    )
    st.altair_chart(chart, width='stretch')

def get_clicked_cluster_id(event, map_df_show: pd.DataFrame) -> int | None:
    if not event or not isinstance(event, dict):
        return None

    sel = event.get("selection") or {}
    objs = sel.get("objects") or {}
    idxs = sel.get("indices") or {}

    if isinstance(objs, dict):
        arr = objs.get("hotspots_scatter")
        if isinstance(arr, list) and arr:
            o = arr[0]
            if isinstance(o, dict) and "cluster_id" in o:
                return int(o["cluster_id"])

    if isinstance(idxs, dict):
        arr = idxs.get("hotspots_scatter")
        if isinstance(arr, list) and arr:
            i = int(arr[0])
            if 0 <= i < len(map_df_show):
                return int(map_df_show.iloc[i]["cluster_id"])

    return None

TYPE_ORDER = [
    "Nightlife" # 夜生活
    "Commuter" # 通勤
    "Business" # 商業活動
    "Weekend/Leisure" # 周末休閒活動
    "Long-trip" # 長距離行程
    "Balanced/Unclear" # 不確定/難以分類
    "Low data" # 資料數據不足
]

def build_cluster_type_profiles(filtered: pd.DataFrame) -> pd.DataFrame:
    df = filtered.copy()
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
    df["hour"] = df["pickup_hour"].dt.hour
    df["dow"] = df["pickup_hour"].dt.dayofweek  # 0 Mon ... 6 Sun
    df["is_weekend"] = df["dow"].isin([5, 6])

    # time masks
    df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4])
    df["is_commute"] = df["hour"].isin([7, 8, 9, 16, 17, 18])
    df["is_weekday_day"] = (~df["is_weekend"]) & df["hour"].between(10, 16)
    df["is_weekend_day"] = (df["is_weekend"]) & df["hour"].between(10, 18)
    df["is_fri_sat_night"] = df["dow"].isin([4, 5]) & df["is_night"]  # Fri(4), Sat(5)
    df["is_evening"] = df["hour"].isin([18, 19, 20, 21])

    # aggregate per cluster
    g = df.groupby("cluster_id", as_index=False).agg(
        total_demand=("demand", "sum"),
        total_revenue=("revenue", "sum"),
        night_demand=("demand", lambda s: float(s[df.loc[s.index, "is_night"]].sum())),
        commute_demand=("demand", lambda s: float(s[df.loc[s.index, "is_commute"]].sum())),
        weekend_demand=("demand", lambda s: float(s[df.loc[s.index, "is_weekend"]].sum())),
        weekday_day_demand=("demand", lambda s: float(s[df.loc[s.index, "is_weekday_day"]].sum())),
        weekend_day_demand=("demand", lambda s: float(s[df.loc[s.index, "is_weekend_day"]].sum())),
        fri_sat_night_demand=("demand", lambda s: float(s[df.loc[s.index, "is_fri_sat_night"]].sum())),
        evening_demand=("demand", lambda s: float(s[df.loc[s.index, "is_evening"]].sum())),
    )

    # shares
    g["avg_rev_per_trip"] = g["total_revenue"] / g["total_demand"].replace(0, np.nan)
    g["night_share"] = g["night_demand"] / g["total_demand"].replace(0, np.nan)
    g["commute_share"] = g["commute_demand"] / g["total_demand"].replace(0, np.nan)
    g["weekend_share"] = g["weekend_demand"] / g["total_demand"].replace(0, np.nan)
    g["weekday_day_share"] = g["weekday_day_demand"] / g["total_demand"].replace(0, np.nan)
    g["weekend_day_share"] = g["weekend_day_demand"] / g["total_demand"].replace(0, np.nan)
    g["fri_sat_night_share"] = g["fri_sat_night_demand"] / g["total_demand"].replace(0, np.nan)
    g["evening_share"] = g["evening_demand"] / g["total_demand"].replace(0, np.nan)

    # peak hour/dow (optional but useful)
    by_hour = df.groupby(["cluster_id", "hour"], as_index=False)["demand"].sum()
    peak_hour = by_hour.sort_values(["cluster_id", "demand"], ascending=[True, False]).drop_duplicates("cluster_id")
    peak_hour = peak_hour.rename(columns={"hour": "peak_hour"})[["cluster_id", "peak_hour"]]

    by_dow = df.groupby(["cluster_id", "dow"], as_index=False)["demand"].sum()
    peak_dow = by_dow.sort_values(["cluster_id", "demand"], ascending=[True, False]).drop_duplicates("cluster_id")
    peak_dow = peak_dow.rename(columns={"dow": "peak_dow"})[["cluster_id", "peak_dow"]]

    g = g.merge(peak_hour, on="cluster_id", how="left").merge(peak_dow, on="cluster_id", how="left")

    # ---- feature -> percentile score (0~1) ----
    g["feat_nightlife"] = g[["night_share", "fri_sat_night_share"]].max(axis=1).fillna(0.0)
    g["feat_commuter"] = g["commute_share"].fillna(0.0)
    g["feat_business"] = (g["weekday_day_share"] - 0.50 * g["weekend_share"]).fillna(0.0)
    g["feat_weekend"] = (g["weekend_share"] + 0.35 * g["weekend_day_share"]).fillna(0.0)
    g["feat_longtrip"] = g["avg_rev_per_trip"].fillna(g["avg_rev_per_trip"].median())

    g["score_nightlife"] = _pct_rank(g["feat_nightlife"])
    g["score_commuter"]  = _pct_rank(g["feat_commuter"])
    g["score_business"]  = _pct_rank(g["feat_business"])
    g["score_weekend"]   = _pct_rank(g["feat_weekend"])
    g["score_longtrip"]  = _pct_rank(g["feat_longtrip"])
    g["afterwork_pctl"] = _pct_rank(g["evening_share"].fillna(0))


    def classify_from_scores(r):
        if (pd.isna(r.total_demand)) or (r.total_demand < 2000):
            return "Low data", 0.0, "insufficient demand", {}

        scores = {
            "Nightlife": float(r.score_nightlife),
            "Commuter": float(r.score_commuter),
            "Business": float(r.score_business),
            "Weekend/Leisure": float(r.score_weekend),
            "Long-trip": float(r.score_longtrip),
        }
        probs = _softmax(scores, temperature=0.45)

        top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        (t1, p1), (t2, p2) = top[0], top[1]
        margin = p1 - p2

        # Balanced/Uncertain
        if p1 < 0.28 or margin < 0.03:
            reason = f"unclear (p1={t1}:{p1:.2f}, p2={t2}:{p2:.2f})"
            conf = float(np.clip(0.35 + (0.28 - p1) * 0.9, 0.30, 0.70))
            return "Balanced/Uncertain", conf, reason, probs

        reason = f"top={t1}:{p1:.2f}, 2nd={t2}:{p2:.2f}, margin={margin:.2f}"
        return t1, float(p1), reason, probs

    out = []
    for r in g.itertuples(index=False):
        t, conf, reason, probs = classify_from_scores(r)
        out.append((
            int(r.cluster_id),
            t,
            float(conf),
            reason,
            float(r.night_share or 0),
            float(r.commute_share or 0),
            float(r.weekend_share or 0),
            float(r.weekday_day_share or 0),
            float(r.evening_share or 0),
            float(r.avg_rev_per_trip) if not pd.isna(r.avg_rev_per_trip) else np.nan,
            float(r.score_nightlife),
            float(r.score_commuter),
            float(r.score_business),
            float(r.score_weekend),
            float(r.score_longtrip),
            float(r.afterwork_pctl),
            float(probs.get("Nightlife", 0)),
            float(probs.get("Commuter", 0)),
            float(probs.get("Business", 0)),
            float(probs.get("Weekend/Leisure", 0)),
            float(probs.get("Long-trip", 0)),
            int(r.peak_hour) if not pd.isna(r.peak_hour) else None,
            int(r.peak_dow) if not pd.isna(r.peak_dow) else None,
            int(r.total_demand),
            float(r.total_revenue),
        ))

    prof = pd.DataFrame(out, columns=[
        "cluster_id","zone_type","type_conf","type_reason",
        "night_share","commute_share","weekend_share","weekday_day_share","evening_share","avg_rev_per_trip",
        "score_nightlife","score_commuter","score_business","score_weekend","score_longtrip", "afterwork_pcl",
        "p_nightlife","p_commuter","p_business","p_weekend","p_longtrip",
        "peak_hour","peak_dow","total_demand","total_revenue",
    ])
    
    # 強制轉型確保輸出
    prof["zone_type"] = prof["zone_type"].fillna("Unknown").astype(str)
    prof["type_reason"] = prof["type_reason"].fillna("").astype(str)

    num_cols = [
        "night_share","commute_share","weekend_share","weekday_day_share","evening_share","avg_rev_per_trip",
        "score_nightlife","score_commuter","score_business","score_weekend","score_longtrip", "afterwork_pcl",
        "p_nightlife","p_commuter","p_business","p_weekend","p_longtrip",
        "type_conf","total_revenue"
    ]
    for c in num_cols:
        if c in prof.columns:
            prof[c] = pd.to_numeric(prof[c], errors="coerce")

    prof["total_demand"] = pd.to_numeric(prof["total_demand"], errors="coerce").fillna(0).astype(int)

    return prof

# -------------------------
# Tabs
# -------------------------
def render_overview_tab(filtered: pd.DataFrame, summary: pd.DataFrame):
    st.subheader("Overview(總覽)")

    total_demand = int(summary["demand"].sum())
    total_revenue = float(summary["revenue"].sum())
    avg_rev_trip = total_revenue / total_demand if total_demand > 0 else np.nan
    active_clusters = int((summary["demand"] > 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total demand(趟數)", f"{total_demand:,}")
    c2.metric("Total revenue(營業額)", f"${total_revenue:,.0f}")
    c3.metric("Avg revenue / trip", "-" if pd.isna(avg_rev_trip) else f"${avg_rev_trip:,.2f}")
    c4.metric("Active clusters", f"{active_clusters:,}")

    st.divider()
    st.subheader("全體日趨勢(Daily + MA7)")

    overall = filtered.groupby("pickup_hour", as_index=False).agg(
        demand=("demand", "sum"),
        revenue=("revenue", "sum"),
    ).sort_values("pickup_hour")

    overall = overall.set_index("pickup_hour")
    daily = overall.resample("D").sum()
    daily["demand_ma7"] = daily["demand"].rolling(7, min_periods=3).mean()
    daily["revenue_ma7"] = daily["revenue"].rolling(7, min_periods=3).mean()

    a, b = st.columns(2)
    with a:
        st.caption("Daily demand + 7-day MA")
        st.line_chart(daily[["demand", "demand_ma7"]], width='stretch')
    with b:
        st.caption("Daily revenue + 7-day MA")
        st.line_chart(daily[["revenue", "revenue_ma7"]], width='stretch')

# -------------------------
# for hotspots 'zone_type' color
# -------------------------

ZONE_TYPE_COLORS = {
    "Nightlife": (255, 0, 160),         # 桃紅/紫
    "Commuter": (0, 200, 255),          # 青藍
    "Business": (255, 190, 0),          # 琥珀黃
    "Weekend/Leisure": (0, 255, 140),   # 綠
    "Long-trip": (170, 170, 255),       # 淡紫
    "Balanced/Uncertain": (180, 200, 220), # 銀灰
    "Low data": (130, 130, 130),        # 深灰
}

def _clamp255(x: float) -> int:
    return int(max(0, min(255, round(x))))

def color_for_zone_type(zone_type: str) -> tuple[int, int, int]:
    if zone_type is None:
        return ZONE_TYPE_COLORS["Balanced/Uncertain"]
    return ZONE_TYPE_COLORS.get(str(zone_type), ZONE_TYPE_COLORS["Balanced/Uncertain"])


def render_hotspots_tab(
    exp: str,
    map_df_show: pd.DataFrame,     # 地圖/表格實際顯示的集合(Top-K 或 All)
    map_df_all: pd.DataFrame,      # 全部 cluster(用來算 share、總量)
    selected_cluster: int,
    metric: str,
    topk: int,
    bubble_min: int,
    bubble_max: int,
    show_rank_labels: bool,
    show_only_topk: bool
):
    map_df_show = map_df_show.reset_index(drop=True)

    if map_df_show.empty:
        st.warning("目前顯示集合為空(可能日期區間太小或資料被清理掉)")
        return
    if map_df_all.empty:
        st.warning("全體集合為空(請確認資料)")
        return

    # ---- Map ----
    layers = build_scatter_layers(
        map_show=map_df_show,
        selected_cluster=selected_cluster,
        metric=metric,
        bubble_min=bubble_min,
        bubble_max=bubble_max,
        show_rank_labels=show_rank_labels,
    )

    view = pdk.ViewState(latitude=40.75, longitude=-73.98, zoom=10)

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view,
        tooltip={
            "text": "點泡泡可切換 cluster\n\n"
                    "Type: {zone_type} (conf={type_conf})\n"
                    "cluster {cluster_id}\n"
                    "demand: {demand_str}\n"
                    "revenue: {revenue_str}\n"
                    "avg_rev/trip: {avg_str}"
        },
    )

    left, right = st.columns([1.2, 1.0])

    with left:
        st.subheader(f"Hotspots(熱點)｜exp: {exp}｜排序/泡泡依 {metric}")
        try:
            event = st.pydeck_chart(
                deck,
                key="hotspots_map",
                width="stretch",
                on_select="rerun",
                selection_mode="single-object",
            )
        except TypeError:
            st.pydeck_chart(deck, width='stretch',)
            st.info("Oh No! 目前 Streamlit 版本尚未支援 pydeck 點選事件")
            event = None
        
        # 解析點到哪一個cluster
        clicked_cluster = get_clicked_cluster_id(event, map_df_show)
        print(clicked_cluster)
        if clicked_cluster is not None and st.session_state.get("selected_cluster_id") != clicked_cluster:
            st.session_state["pending_cluster_id"] = int(clicked_cluster)
            st.rerun()

    with right:
        legend_html = "<div style='display:flex;flex-wrap:wrap;gap:10px;align-items:center;'>"
        for k, (r,g,b) in ZONE_TYPE_COLORS.items():
            legend_html += (
                f"<div style='display:flex;align-items:center;gap:6px;'>"
                f"<span style='width:12px;height:12px;border-radius:3px;background:rgb({r},{g},{b});display:inline-block;'></span>"
                f"<span style='font-size:12px;color:#cbd5e1;'>{k}</span>"
                f"</div>"
            )
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

        st.subheader("熱區排行榜(Highlight目前選取cluster)")
        show_cols = ["rank", "cluster_id", "zone_type", "type_conf", "demand", "revenue", "avg_rev_per_trip", "lon", "lat"]
        df_show = map_df_show[show_cols].copy()
        def highlight_row(row):
            if int(row["cluster_id"]) == int(selected_cluster):
                return ["background-color: rgba(255,120,0,0.25); font-weight: 700"] * len(row)
            return [""] * len(row)
        
        st.dataframe(
            df_show.style.apply(highlight_row, axis=1).format(
                {"revenue": "{:,.0f}", "avg_rev_per_trip": "{:.2f}", "lon": "{:.4f}", "lat": "{:.4f}"}
            ),
            width='stretch',
            height=520,
            hide_index=True,
        )

    st.divider()

    # ---- Share stats (Top-K share is always computed against full population) ----
    total_d = float(map_df_all["demand"].sum())
    total_r = float(map_df_all["revenue"].sum())
    shown_d = float(map_df_show["demand"].sum())
    shown_r = float(map_df_show["revenue"].sum())

    shown_d_share = (shown_d / total_d) if total_d > 0 else np.nan
    shown_r_share = (shown_r / total_r) if total_r > 0 else np.nan

    top_part = map_df_all.head(topk)
    top_d_share = (float(top_part["demand"].sum()) / total_d) if total_d > 0 else np.nan
    top_r_share = (float(top_part["revenue"].sum()) / total_r) if total_r > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("顯示筆數(地圖/表格)", f"{len(map_df_show):,}")
    
    label_prefix = f"Top-{topk}" if show_only_topk else "顯示集合"
    c2.metric(f"{label_prefix} demand 佔比", "-" if pd.isna(shown_d_share) else f"{shown_d_share*100:.1f}%")
    c3.metric(f"{label_prefix} revenue 佔比", "-" if pd.isna(shown_r_share) else f"{shown_r_share*100:.1f}%")
    
    if not show_only_topk:
        c4.metric(f"Top-{topk} demand 佔比(參考)", "-" if pd.isna(top_d_share) else f"{top_d_share*100:.1f}%")
    else:
        c4.metric(f"Top-{topk} revenue 佔比(參考)", "-" if pd.isna(top_r_share) else f"{top_r_share*100:.1f}%")

def render_zone_type_info(profiles: pd.DataFrame, cluster_id: int):
    row = profiles.loc[profiles["cluster_id"] == int(cluster_id)]
    if row.empty:
        st.info("此 cluster 沒有 zone_type 推論結果。")
        return

    r = row.iloc[0].to_dict()

    st.markdown(
        f"### Zone Type：**{r.get('zone_type','Unknown')}**  (confidence={float(r.get('type_conf',0.0)):.2f})\n"
        f"- reason: `{r.get('type_reason','')}`"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("night_share", f"{float(r.get('night_share',0.0)):.0%}")
    c2.metric("commute_share", f"{float(r.get('commute_share',0.0)):.0%}")
    c3.metric("weekday_day_share", f"{float(r.get('weekday_day_share',0.0)):.0%}")
    c4.metric("weekend_share", f"{float(r.get('weekend_share',0.0)):.0%}")
    c5.metric("evening_share", f"{float(r.get('evening_share',0.0)):.0%}")

    tags = []
    if float(r.get("afterwork_pctl", 0)) >= 0.85:
        tags.append("Afterwork-heavy")
    if float(r.get("night_share", 0)) >= 0.25:
        tags.append("Night-heavy")

    if tags:
        st.caption("Tags: " + ", ".join(tags))


    prob_df = pd.DataFrame({
        "type": ["Nightlife","Commuter","Business","Weekend/Leisure","Long-trip"],
        "prob": [
            r.get("p_nightlife", np.nan),
            r.get("p_commuter", np.nan),
            r.get("p_business", np.nan),
            r.get("p_weekend", np.nan),
            r.get("p_longtrip", np.nan),
        ],
        "score(pctl)": [
            r.get("score_nightlife", np.nan),
            r.get("score_commuter", np.nan),
            r.get("score_business", np.nan),
            r.get("score_weekend", np.nan),
            r.get("score_longtrip", np.nan),
        ],
    })

    prob_df["prob"] = pd.to_numeric(prob_df["prob"], errors="coerce")
    prob_df["score(pctl)"] = pd.to_numeric(prob_df["score(pctl)"], errors="coerce")

    prob_df = prob_df.sort_values("prob", ascending=False)

    st.caption("分類依據(prob=softmax 機率；score=特徵在所有 clusters 的百分位 0~1)")
    st.dataframe(
        prob_df.style.format({"prob": "{:.2f}", "score(pctl)": "{:.2f}"}),
        width='stretch',
        height=210,
        hide_index=True,
    )
    
    prob_plot = prob_df.copy()
    prob_plot["prob"] = pd.to_numeric(prob_plot["prob"], errors="coerce")

    bar = (
        alt.Chart(prob_plot)
        .mark_bar()
        .encode(
            x=alt.X("type:N", title="Type", sort="-y", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("prob:Q", title="Probabilities", axis=alt.Axis(labelAngle=0)),
        )
    )

    text = (
        alt.Chart(prob_plot)
        .mark_text(dy=-10, color="white", clip=False)
        .encode(
            x=alt.X("type:N", sort="-y"),
            y="prob:Q",
            text=alt.Text("prob:Q", format=".2f"),
        )
    )

    st.altair_chart(bar + text, width='stretch')

    #st.bar_chart(prob_df.set_index("type")["prob"])

    
def render_trends_tab(filtered: pd.DataFrame, selected_cluster: int, profiles: pd.DataFrame):
    render_zone_type_info(profiles, selected_cluster)
    st.divider()
    st.subheader(f"Trends(趨勢)｜cluster {selected_cluster}")
    st.divider()

    ts = filtered[filtered["cluster_id"] == selected_cluster].sort_values("pickup_hour")
    if ts.empty:
        st.warning("此日期區間該 cluster 沒有資料。")
        return

    ts = ts.set_index("pickup_hour")
    ts["avg_rev_per_trip"] = ts["revenue"] / ts["demand"].replace(0, np.nan)

    daily = ts.resample("D").sum()
    daily["demand_ma7"] = daily["demand"].rolling(7, min_periods=3).mean()
    daily["revenue_ma7"] = daily["revenue"].rolling(7, min_periods=3).mean()

    st.caption("日趨勢(Daily + MA7)")
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(daily[["demand", "demand_ma7"]], width='stretch')
    with c2:
        st.line_chart(daily[["revenue", "revenue_ma7"]], width='stretch')

    st.divider()
    st.subheader(f"週期輪廓(日/週)｜cluster {selected_cluster}")

    hod_profile = build_hod_profile(ts[["demand", "revenue"]])
    dow_profile = build_dow_profile(ts[["demand", "revenue"]])
    dow_plot = dow_profile.reset_index().rename(columns={"index": "dow"})

    c3, c4 = st.columns(2)
    with c3:
        st.caption("Hour-of-day：Demand")
        st.bar_chart(hod_profile["demand"], width='stretch')
    with c4:
        st.caption("Day-of-week：Demand")
        altair_bar(dow_plot, "dow", "demand", "", sort=DOW_ORDER)

    c5, c6 = st.columns(2)
    with c5:
        st.caption("Hour-of-day：Avg revenue/trip")
        st.bar_chart(hod_profile["avg_rev_per_trip"], width='stretch')
    with c6:
        st.caption("Day-of-week：Avg revenue/trip")
        altair_bar(dow_plot, "dow", "avg_rev_per_trip", "", sort=DOW_ORDER)

    st.divider()

    st.subheader(f"尖峰時段(Top 10)｜cluster {selected_cluster}")
    spikes = ts.reset_index().copy()
    spikes["avg_rev_per_trip"] = spikes["revenue"] / spikes["demand"].replace(0, np.nan)

    top_demand = spikes.sort_values("demand", ascending=False).head(10)[
        ["pickup_hour", "demand", "revenue", "avg_rev_per_trip"]
    ]
    top_avg = spikes.sort_values("avg_rev_per_trip", ascending=False).head(10)[
        ["pickup_hour", "demand", "revenue", "avg_rev_per_trip"]
    ]

    a, b = st.columns(2)
    with a:
        st.caption("Demand 最高的 10 個小時")
        st.dataframe(top_demand, width='stretch', height=320)
    with b:
        st.caption("Avg revenue/trip 最高的 10 個小時")
        st.dataframe(top_avg, width='stretch', height=320)

# -------------------------
# main()
# -------------------------
def main():
    st.set_page_config(page_title="NYC Taxi | Demand & Revenue Dashboard", layout="wide")
    st.title("NYC Yellow Taxi｜需求熱點 & 營業額分析(2016/01-03)")

    # -------------------------
    # Experiments
    # -------------------------
    exps = list_experiments(BASE_FEAT_DIR)
    if not exps:
        st.error(
            "找不到任何 features 結果。請先跑 run_pipeline.py 產生：\n"
            "- data/features/<exp>/zone_hour.parquet\n"
            "- data/features/<exp>/cluster_centers.parquet"
        )
        return

    with st.sidebar:
        st.header("資料來源 / 模型結果")
        exp = st.selectbox("選擇 clustering 實驗(--exp)", exps, index=0)

    zone, centers = load_data(exp)
    min_t, max_t = zone["pickup_hour"].min(), zone["pickup_hour"].max()

    # -------------------------
    # Sidebar filters
    # -------------------------
    with st.sidebar:
        st.header("篩選")
        date_range = st.date_input("日期區間", value=(min_t.date(), max_t.date()))

        st.header("熱度指標")
        metric = st.radio("排序 / 泡泡大小依", ["demand", "revenue", "avg_rev_per_trip"], horizontal=False)

        st.header("地圖顯示")
        show_only_topk = st.checkbox("地圖只顯示 Top-K", value=True)
        topk = st.slider("Top-K", 5, 60, 15)

        st.header("泡泡清晰度")
        bubble_min = st.slider("最小泡泡半徑", 10, 300, 50)
        bubble_max = st.slider("最大泡泡半徑", 300, 1000, 500)
        show_rank_labels = st.checkbox("顯示排名文字", value=True)

    # -------------------------
    # Filter & summaries
    # -------------------------
    filtered, ready = date_filter(zone, date_range)
    if not ready:
        st.info("請先在左側選擇完整的日期區間(起始日與結束日)。")
        st.stop()

    if filtered.empty:
        st.warning("此日期區間沒有資料。請換個日期區間。")
        st.stop()

    @st.cache_data
    def cached_profiles(filtered_df: pd.DataFrame) -> pd.DataFrame:
        return build_cluster_type_profiles(filtered_df)

    profiles = cached_profiles(filtered)
    summary = cluster_summary(filtered)
    
    map_df = make_map_df(summary, centers, metric)
    if map_df.empty:
        st.warning("此日期區間沒有可用的熱區中心點。")
        return

    map_df_all = map_df
    map_df_show = map_df_all.head(topk).copy() if show_only_topk else map_df_all.copy()
    
    map_df_all = map_df_all.merge(profiles[["cluster_id","zone_type","type_conf"]], on="cluster_id", how="left")
    map_df_show = map_df_show.merge(profiles[["cluster_id","zone_type","type_conf"]], on="cluster_id", how="left")

    # -------------------------
    # Cluster options (sync with display mode)
    # -------------------------
    options_ids = (
        map_df_show["cluster_id"].tolist()
        if show_only_topk
        else map_df_all["cluster_id"].tolist()
    )
    if not options_ids:
        st.warning("目前沒有可供選取的熱區。")
        return

    label_map = dict(zip(map_df_all["cluster_id"], map_df_all["label"]))

    # -------------------------
    # Init & repair state
    # -------------------------
    
    # selected_cluster init
    if "selected_cluster_id" not in st.session_state:
        st.session_state["selected_cluster_id"] = int(options_ids[0])
    if "selected_cluster_id_widget" not in st.session_state:
        st.session_state["selected_cluster_id_widget"] = int(st.session_state["selected_cluster_id"])

    # pending_cluster_id用於hotspot選取泡泡時rerun更新使用
    if "pending_cluster_id" not in st.session_state:
        st.session_state["pending_cluster_id"] = None
    # 當有進行選取泡泡時就會去執行以下更新
    if st.session_state["pending_cluster_id"] is not None:
        pc = int(st.session_state["pending_cluster_id"])
        st.session_state["pending_cluster_id"] = None  # clear first

        if pc not in options_ids:
            pc = int(options_ids[0])

        st.session_state["selected_cluster_id"] = pc
        st.session_state["selected_cluster_id_widget"] = pc

    # 左側sidebar有選項變動時 需要同步更新單一熱區的選中cluster(預設第一個)
    if st.session_state["selected_cluster_id"] not in options_ids:
        st.session_state["selected_cluster_id"] = int(options_ids[0])
    if st.session_state["selected_cluster_id_widget"] not in options_ids:
        st.session_state["selected_cluster_id_widget"] = int(st.session_state["selected_cluster_id"])

    # -------------------------
    # Sidebar cluster selector
    # -------------------------
    
    # 
    def sync_widget_to_state():
        st.session_state["selected_cluster_id"] = int(st.session_state["selected_cluster_id_widget"])

    with st.sidebar:
        st.header("查看單一熱區")
        selected_widget_val = st.selectbox(
            "依目前指標排序的熱區列表",
            options_ids,
            key="selected_cluster_id_widget",
            format_func=lambda cid: label_map.get(int(cid), f"cluster {int(cid)}"),
            on_change=sync_widget_to_state,
        )
        if int(selected_widget_val) != int(st.session_state["selected_cluster_id"]):
            st.session_state["selected_cluster_id"] = int(selected_widget_val)
    

    selected_cluster = int(st.session_state["selected_cluster_id"])

    # -------------------------
    # Tabs
    # -------------------------
    tab_overview, tab_hotspots, tab_trends, tab_dispatch, tab_apriori = st.tabs(
        ["Overview(總覽)", "Hotspots(熱點)", "Trends(趨勢)", "Dispatch(計程車調度預測)", "Apriori"]
    )

    with tab_overview:
        render_overview_tab(filtered, summary)

    with tab_hotspots:
        render_hotspots_tab(
            exp=exp,
            map_df_show=map_df_show,
            map_df_all=map_df_all,
            selected_cluster=selected_cluster,
            metric=metric,
            topk=topk,
            bubble_min=bubble_min,
            bubble_max=bubble_max,
            show_rank_labels=show_rank_labels,
            show_only_topk=show_only_topk,
        )

    with tab_trends:
        render_trends_tab(filtered, selected_cluster, profiles)

    with tab_dispatch:
        render_dispatch_tab(
            zone_all=zone, 
            centers=centers,
            profiles=profiles, 
            metric_global=metric,
            safe_log_radius_fn=safe_log_radius,
            color_for_zone_type_fn=color_for_zone_type,
        )
    
    with tab_apriori:
        render_apriori_tab(APR_TX_PATH.as_posix())


if __name__ == "__main__":
    main()
