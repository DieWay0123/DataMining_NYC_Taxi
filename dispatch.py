from __future__ import annotations

from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

@st.cache_data(show_spinner=False)
def compute_dispatch_plan(
    zone_all: pd.DataFrame,
    centers: pd.DataFrame,
    profiles: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon_h: int = 6,
    window_days: int = 28,
) -> pd.DataFrame:
    df = zone_all.copy()
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
    df["hour"] = df["pickup_hour"].dt.hour
    df["dow"] = df["pickup_hour"].dt.dayofweek

    as_of = pd.to_datetime(as_of)

    # ---- Baseline training window ----
    train_start = as_of - pd.Timedelta(days=window_days)
    train = df[(df["pickup_hour"] < as_of) & (df["pickup_hour"] >= train_start)].copy()
    if train.empty:
        train = df[df["pickup_hour"] < as_of].copy()

    agg = train.groupby(["cluster_id", "dow", "hour"], as_index=False).agg(
        mean_demand=("demand", "mean"),
        mean_revenue=("revenue", "mean"),
    )
    agg_c = train.groupby("cluster_id", as_index=False).agg(
        fb_demand=("demand", "mean"),
        fb_revenue=("revenue", "mean"),
    )

    # ---- Targets: next 1..horizon hours ----
    targets = pd.DataFrame({"target_time": [as_of + timedelta(hours=i) for i in range(1, horizon_h + 1)]})
    targets["dow"] = pd.to_datetime(targets["target_time"]).dt.dayofweek
    targets["hour"] = pd.to_datetime(targets["target_time"]).dt.hour

    clusters = pd.DataFrame({"cluster_id": centers["cluster_id"].astype(int).unique()})
    targets["key"] = 1
    clusters["key"] = 1
    grid = targets.merge(clusters, on="key").drop(columns=["key"])

    grid = grid.merge(agg, on=["cluster_id", "dow", "hour"], how="left")
    grid = grid.merge(agg_c, on="cluster_id", how="left")

    grid["pred_demand_1h"] = pd.to_numeric(grid["mean_demand"], errors="coerce").fillna(grid["fb_demand"]).fillna(0.0)
    grid["pred_revenue_1h"] = pd.to_numeric(grid["mean_revenue"], errors="coerce").fillna(grid["fb_revenue"]).fillna(0.0)

    # peak hour (for action text)
    idx = grid.groupby("cluster_id")["pred_demand_1h"].idxmax()
    peak = grid.loc[idx, ["cluster_id", "target_time", "hour", "dow", "pred_demand_1h"]].rename(
        columns={"target_time": "peak_time", "hour": "peak_hour", "dow": "peak_dow", "pred_demand_1h": "peak_pred_1h"}
    )

    plan = grid.groupby("cluster_id", as_index=False).agg(
        pred_demand=("pred_demand_1h", "sum"),
        pred_revenue=("pred_revenue_1h", "sum"),
    ).merge(peak, on="cluster_id", how="left")

    plan["pred_avg_rev_per_trip"] = plan["pred_revenue"] / plan["pred_demand"].replace(0, np.nan)

    # ---- merge centers & profiles ----
    plan = plan.merge(centers[["cluster_id", "lon", "lat"]], on="cluster_id", how="left")
    plan = plan.merge(profiles[["cluster_id", "zone_type", "type_conf"]], on="cluster_id", how="left")
    plan["zone_type"] = plan["zone_type"].fillna("Unknown").astype(str)
    plan["type_conf"] = pd.to_numeric(plan["type_conf"], errors="coerce").fillna(0.0)

    # ---- action text ----
    commute_hours = {7, 8, 9, 16, 17, 18}
    nightlife_hours = {22, 23, 0, 1, 2}

    def make_action(r):
        zt = r["zone_type"]
        ph = int(r["peak_hour"]) if pd.notna(r["peak_hour"]) else None
        pdow = int(r["peak_dow"]) if pd.notna(r["peak_dow"]) else None
        weekend = (pdow in [5, 6]) if pdow is not None else False

        if zt == "Nightlife":
            return "夜生活尖峰：建議增派到深夜時段" if ph in nightlife_hours else "夜間需求偏高：留意晚間調度"
        if zt == "Commuter":
            return "通勤尖峰：建議提前佈署(早晚高峰)" if ph in commute_hours else "通勤型區域：觀察上下班時段"
        if zt == "Business":
            return "商務區：平日白天較活躍"
        if zt == "Weekend/Leisure":
            return "週末休閒：週六日常較熱" if weekend else "休閒型：留意假日"
        if zt == "Long-trip":
            return "長途/高客單：適合衝營收"
        if zt == "Balanced/Uncertain":
            return "型態不明：以即時熱點為主"
        return "—"

    plan["action"] = plan.apply(make_action, axis=1)

    # display rounding
    plan["pred_demand"] = pd.to_numeric(plan["pred_demand"], errors="coerce").fillna(0).round(0)
    plan["pred_revenue"] = pd.to_numeric(plan["pred_revenue"], errors="coerce").fillna(0).round(0)
    plan["pred_avg_rev_per_trip"] = (plan["pred_avg_rev_per_trip"]).round(2)

    return plan

def render_dispatch_tab(
    zone_all: pd.DataFrame,
    centers: pd.DataFrame,
    profiles: pd.DataFrame,
    metric_global: str,
    safe_log_radius_fn,
    color_for_zone_type_fn,
):
    st.subheader("Dispatch(調度建議)")

    metric = st.radio(
        "以哪個指標做 Top-N 推薦？",
        ["pred_demand", "pred_revenue", "pred_avg_rev_per_trip"],
        horizontal=True,
        index=0 if metric_global == "demand" else (1 if metric_global == "revenue" else 2),
    )

    hours = pd.to_datetime(zone_all["pickup_hour"]).sort_values().unique()
    if len(hours) == 0:
        st.warning("zone_all 沒有 pickup_hour。")
        return

    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
    with c1:
        as_of = st.selectbox("As-of(以此時間為基準預測)", hours, index=max(0, len(hours) - 25))
    with c2:
        horizon_h = st.slider("預測未來幾小時", 1, 12, 6)
    with c3:
        window_days = st.slider("Baseline 視窗(天)", 7, 60, 28)
    with c4:
        topn = st.slider("Top-N 推薦", 5, 50, 15)
    
    plan = compute_dispatch_plan(
        zone_all=zone_all,
        centers=centers,
        profiles=profiles,
        as_of=pd.to_datetime(as_of),
        horizon_h=int(horizon_h),
        window_days=int(window_days),
    )


    plan = plan.sort_values(metric, ascending=False).head(topn).copy()
    plan["rank"] = np.arange(1, len(plan) + 1)
    plan["peak_time_str"] = pd.to_datetime(plan["peak_time"]).dt.strftime("%Y-%m-%d %H:%M")

    show_cols = [
        "rank","cluster_id","zone_type","type_conf",
        "pred_demand","pred_revenue","pred_avg_rev_per_trip",
        "peak_time_str","action",
    ]

    st.dataframe(
        plan[show_cols].style.format({
            "type_conf": "{:.2f}",
            "pred_demand": "{:,.0f}",
            "pred_revenue": "${:,.0f}",
            "pred_avg_rev_per_trip": "${:,.2f}",
        }),
        width='stretch',
        height=420,
    )

    st.caption("pred_* 為 baseline 預測(過去 N 天同 dow+hour 平均)；peak_time 為未來 H 小時內預測 demand 最強的時段。")

    # Map
    map_df = plan.copy()

    # 清理座標型態
    map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
    map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
    map_df = map_df.dropna(subset=["lon", "lat"]).copy()

    if map_df.empty:
        st.warning("Dispatch 地圖沒有可用座標(lon/lat 全是空)。請檢查 centers merge 是否成功。")
        return

    # Bubble size
    map_df["bubble_value"] = pd.to_numeric(map_df[metric], errors="coerce").fillna(0.0)

    v = map_df["bubble_value"].to_numpy(dtype=float)
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    denom = (vmax - vmin) if vmax > vmin else 1.0

    # 8~40px
    map_df["radius_px"] = 8 + 32 * np.sqrt((map_df["bubble_value"] - vmin) / denom)

    # rank text
    map_df["rank_str"] = map_df["rank"].astype(int).astype(str)

    # Bubble 顏色
    def fill(row):
        r, g, b = color_for_zone_type_fn(row["zone_type"])
        return [int(r), int(g), int(b), 190]

    map_df["fill_color"] = map_df.apply(fill, axis=1)

    # 用 Top-N 的中心
    view_lat = float(map_df["lat"].mean())
    view_lon = float(map_df["lon"].mean())

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=map_df.sort_values("radius_px", ascending=False),  # 大的先畫，避免蓋住小的
        id="dispatch_scatter",
        get_position="[lon, lat]",
        radius_units="pixels",
        get_radius="radius_px",
        get_fill_color="fill_color",
        stroked=True,
        get_line_color=[255, 255, 255, 220],
        line_width_min_pixels=2,
        pickable=True,
        auto_highlight=True,
    )

    text = pdk.Layer(
        "TextLayer",
        data=map_df,
        id="dispatch_text",
        get_position="[lon, lat]",
        get_text="rank_str",
        get_size=18,
        get_color=[255, 255, 255, 230],
        get_text_anchor="'middle'",
        get_alignment_baseline="'center'",
        pickable=False,  # 避免文字層干擾點選泡泡
    )

    deck = pdk.Deck(
        layers=[scatter, text],
        initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=11, pitch=0),
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={
            "text": "rank #{rank}\ncluster {cluster_id}\nType: {zone_type} (conf={type_conf})\n"
                    f"{metric}: " + "{" + metric + "}\n"
                    "peak_time: {peak_time_str}\n"
                    "action: {action}"
        },
    )

    st.pydeck_chart(deck, width='stretch')
