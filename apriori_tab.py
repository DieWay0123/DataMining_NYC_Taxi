from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from rules_engine import mine_rules_from_transactions, AprioriParams


# -------------------------
# Utilities
# -------------------------
def _list_tx_candidates() -> list[str]:
    # 資料集合多選：從 artifacts 底下找交易表 parquet
    # 資料單位: cluster每小時所有行程為一個單位 EX: 10個cluster 24hr就有240筆資料去做apriori
    base = Path("artifacts")
    if not base.exists():
        return []

    files = []
    files += list(base.glob("cluster_hour_tx*.parquet"))
    files += [p for p in base.glob("*.parquet") if "tx" in p.name.lower()]

    uniq = sorted({str(p.as_posix()) for p in files})
    return uniq


@st.cache_data(show_spinner=False)
def _load_one_tx(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _load_many_txs(paths: tuple[str, ...]) -> pd.DataFrame:
    dfs = [_load_one_tx(p) for p in paths]
    if len(dfs) == 1:
        return dfs[0]
    return pd.concat(dfs, axis=0, ignore_index=True)


def _build_items_from_row(row, selected_fields: list[str], include_cluster_item: bool) -> list[str]:
    """
    交易表一列 -> items list
    selected_fields: 使用者在 UI 選的欄位
    """
    items: list[str] = []
    if include_cluster_item:
        items.append(f"CLUSTER={int(row.cluster_id)}")

    for col in selected_fields:
        val = getattr(row, col, None)
        if val is None:
            continue
        s = str(val)
        if s == "nan":
            continue
        items.append(f"{col}={s}")
    return items


def _rules_filter_by_text(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    if not keyword.strip():
        return df
    k = keyword.strip().lower()
    return df[
        df["antecedents"].str.lower().str.contains(k, na=False)
        | df["consequents"].str.lower().str.contains(k, na=False)
    ]


# -------------------------
# Main Tab
# -------------------------
def render_apriori_tab(default_tx: str = "artifacts/cluster_hour_tx.parquet"):
    # session init
    if "apriori_rules" not in st.session_state:
        st.session_state.apriori_rules = None
    if "apriori_toast" not in st.session_state:
        st.session_state.apriori_toast = None
    if "apriori_last_run_meta" not in st.session_state:
        st.session_state.apriori_last_run_meta = None

    if st.session_state.apriori_toast:
        st.toast(st.session_state.apriori_toast, icon="🔥")
        st.session_state.apriori_toast = None

    left, right = st.columns([1, 3], vertical_alignment="top")

    # -------------------------
    # LEFT: options + setting
    # -------------------------
    with left:
        form = st.form("apriori_form_keep_ui")

        candidates = _list_tx_candidates()
        if default_tx not in candidates and Path(default_tx).exists():
            candidates = [default_tx] + candidates

        dataset_paths = form.multiselect(
            "選擇你要使用的資料集合(交易表 .parquet，可多選)",
            options=candidates,
            default=[default_tx] if Path(default_tx).exists() else [],
            key="apriori_dataset_paths",
        )

        candidate_cols = [
            "Pickup_Time",
            "Tip_Level",
            "Passenger",
            "TripDistance",
            "PaymentType",
            "TotalAmount",
            "FareAmount",
        ]
        cols = form.multiselect(
            "選擇你要使用的欄位(可多選)",
            options=candidate_cols,
            default=candidate_cols,
            key="apriori_cols",
        )

        min_support = form.slider(
            "Min support",
            min_value=0.005,
            max_value=0.30,
            value=0.05,
            step=0.005,
            key="apriori_min_support",
        )

        metric = form.selectbox(
            "Metric",
            options=["lift"],
            index=0,
            key="apriori_metric",
        )

        min_threshold = form.slider(
            "Min threshold",
            min_value=1.0,
            max_value=6.0,
            value=1.1,
            step=0.1,
            key="apriori_min_threshold",
        )

        head_n = form.slider(
            "Head n",
            min_value=1,
            max_value=300,
            value=80,
            step=1,
            key="apriori_head_n",
        )

        with form.expander("進階設定", expanded=False):
            max_len = st.slider("Max itemset length", 2, 5, 3, 1, key="apriori_max_len")
            sort_by = st.selectbox("排序依據", ["confidence", "lift", "support"], index=0, key="apriori_sort_by")
            include_cluster_item = st.checkbox(
                "把 CLUSTER 也當作 item(可挖 CLUSTER=51 -> ...)",
                value=False,
                key="apriori_include_cluster_item",
            )
            min_trips = st.slider(
                "每筆交易最少 trips",
                1, 300, 10, 1,
                key="apriori_min_trips",
            )

        run = form.form_submit_button("開始分析！", type="primary")

    # -------------------------
    # RIGHT: run + display
    # -------------------------
    with right:
        if run:
            if not dataset_paths:
                st.session_state.apriori_toast = "請選擇至少一個資料集合(交易表)"
                st.rerun()

            if not cols:
                st.session_state.apriori_toast = "請選擇至少一個欄位"
                st.rerun()

            # load tx (multi)
            try:
                tx = _load_many_txs(tuple(dataset_paths))
            except Exception as e:
                st.error(f"讀取交易表失敗：{e}")
                st.stop()

            # require columns
            need = {"cluster_id", "trips"}
            miss = need - set(tx.columns)
            if miss:
                st.error(f"交易表缺少必要欄位：{sorted(miss)}")
                st.stop()

            tx = tx.copy()
            tx = tx[tx["trips"] >= int(min_trips)].copy()

            # intersect cols
            available_cols = [c for c in cols if c in tx.columns]
            if not available_cols:
                st.error("你選的欄位在交易表中都不存在(可能 dataset 不含該欄位)。")
                st.stop()

            st.caption(f"本次交易筆數(cluster×hour): {len(tx):,}")

            # build items
            items = []
            for r in tx.itertuples(index=False):
                items.append(_build_items_from_row(r, available_cols, include_cluster_item))

            params = AprioriParams(
                min_support=float(min_support),
                metric=str(metric),
                min_threshold=float(min_threshold),
                head_n=int(head_n),
                max_len=int(max_len),
                sort_by=str(sort_by),
            )

            with st.spinner("Running Apriori..."):
                rules = mine_rules_from_transactions(items, params)

            st.session_state.apriori_rules = rules
            st.session_state.apriori_last_run_meta = {
                "datasets": dataset_paths,
                "cols": available_cols,
                "include_cluster_item": include_cluster_item,
                "min_support": min_support,
                "metric": metric,
                "min_threshold": min_threshold,
                "head_n": head_n,
                "max_len": max_len,
                "sort_by": sort_by,
                "min_trips": int(min_trips),
                "rows": int(len(tx)),
            }

        # ---- Always render from session_state ----
        meta = st.session_state.get("apriori_last_run_meta")
        rules = st.session_state.get("apriori_rules")

        if rules is None:
            st.info("請在左側設定資料集合/欄位與參數後，按「開始分析！」")
            return

        if rules.empty:
            st.warning("沒有挖到規則：試著降低 min_support / min_threshold 或放寬篩選。")
            return

        if meta:
            st.write(
                f"**分析完成！**  "
                f"Min support={meta['min_support']} | Metric={meta['metric']} | Min threshold={meta['min_threshold']} | "
                f"Head n={meta['head_n']} | max_len={meta['max_len']} | rows={meta['rows']:,}"
            )
            st.caption(f"Datasets: {', '.join(meta['datasets'])}")
            st.caption(f"Cols: {', '.join(meta['cols'])}" + (" + CLUSTER" if meta.get("include_cluster_item") else ""))

        st.markdown("### 規則結果")

        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            keyword = st.text_input(
                "關鍵字篩選(例：HighTip / Cash / Evening)",
                value="",
                key="apriori_rules_keyword",
            )
        with f2:
            min_conf = st.slider("confidence ≥", 0.0, 1.0, 0.0, 0.05, key="apriori_rules_min_conf")
        with f3:
            min_lift = st.slider("lift ≥", 1.0, 12.0, 1.0, 0.1, key="apriori_rules_min_lift")

        rules2 = rules.copy()
        rules2 = _rules_filter_by_text(rules2, keyword)
        rules2 = rules2[(rules2["confidence"] >= float(min_conf)) & (rules2["lift"] >= float(min_lift))]

        rules2 = rules2.reset_index(drop=True)
        rules2.insert(0, "row_no", range(1, len(rules2) + 1))

        st.dataframe(rules2, width='stretch', height=520)

        st.download_button(
            "下載規則 CSV",
            data=rules2.to_csv(index=False).encode("utf-8-sig"),
            file_name="apriori_rules_filtered.csv",
            mime="text/csv",
            key="apriori_download_rules",
        )