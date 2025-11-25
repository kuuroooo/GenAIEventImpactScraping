import os
import json
import pandas as pd
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # 使用无界面后端，方便在任何环境下保存图片
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================

DATA_FILE = "mastodon_data.json"
BASELINE_DAYS = 7   # 事件前 7 天
POST_DAYS = 7       # 事件后 7 天

# 输出文件名（表格）
EVENT_SUMMARY_CSV = "event_level_summary.csv"
INSTANCE_SUMMARY_CSV = "instance_level_summary.csv"
HETEROGENEITY_CSV = "event_instance_heterogeneity.csv"

# 输出图像文件
FIG_EVENT_POLARITY = "fig_event_polarity_baseline_vs_post.png"
FIG_EVENT_VOLUME = "fig_event_volume_baseline_vs_post.png"
FIG_YEAR_TRENDS = "fig_year_trends.png"
FIG_INSTANCE_HETEROGENEITY = "fig_instance_heterogeneity.png"


# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================

def load_data(data_file: str) -> pd.DataFrame:
    """Load line-delimited JSON into a pandas DataFrame."""
    if not os.path.exists(data_file):
        print(f"[ERROR] File '{data_file}' not found. Run the scraper first.")
        return pd.DataFrame()

    print(f"[INFO] Loading data from '{data_file}'...")
    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines but keep going
                continue

    df = pd.DataFrame(data)
    print(f"[INFO] Loaded {len(df)} rows.")
    return df


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    解析 event_date 和 created_at，统一到同一个时区后去掉时区信息，
    计算 delta_days，并打上 baseline/post/event_day/outside 标签。
    """
    df = df.copy()

    # 1. 统一解析为 tz-aware（UTC）
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce", utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    # 2. 去掉时区信息，变成 tz-naive，避免 tz-naive / tz-aware 冲突
    df["event_date"] = df["event_date"].dt.tz_convert(None)
    df["created_at"] = df["created_at"].dt.tz_convert(None)

    # 3. 丢掉时间解析失败的行
    before = len(df)
    df = df.dropna(subset=["event_date", "created_at"])
    after = len(df)
    if after < before:
        print(f"[WARN] Dropped {before - after} rows due to invalid dates.")

    # 4. 计算按天的差值
    df["delta_days"] = (
        df["created_at"].dt.normalize() - df["event_date"].dt.normalize()
    ).dt.days

    # 5. 打 period 标签
    def label_period(delta: int) -> str:
        if -BASELINE_DAYS <= delta <= -1:
            return "baseline"
        elif 1 <= delta <= POST_DAYS:
            return "post"
        elif delta == 0:
            return "event_day"
        else:
            return "outside"

    df["period"] = df["delta_days"].apply(label_period)

    # 6. 一些辅助列
    df["event_name"] = df["event"]
    df["event_year"] = df["event_date"].dt.year

    return df


# ==========================================
# EVENT-LEVEL ANALYSIS
# ==========================================

def compute_event_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute baseline vs post metrics at the event level."""
    df_bp = df[df["period"].isin(["baseline", "post"])].copy()
    if df_bp.empty:
        print("[WARN] No data in baseline/post windows.")
        return pd.DataFrame()

    base_group_cols = ["event_name", "event_date", "event_year", "period"]

    # 帖子数 & 平均情绪
    agg_basic = df_bp.groupby(base_group_cols)["polarity"].agg(
        n_posts="count",
        mean_polarity="mean"
    ).reset_index()

    # 正/中/负分布
    sent_counts = (
        df_bp.groupby(base_group_cols + ["sentiment_class"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for c in ["Negative", "Neutral", "Positive"]:
        if c not in sent_counts.columns:
            sent_counts[c] = 0

    sent_counts["total"] = (
        sent_counts["Negative"] + sent_counts["Neutral"] + sent_counts["Positive"]
    ).replace(0, pd.NA)

    for c in ["Negative", "Neutral", "Positive"]:
        sent_counts[f"{c.lower()}_share"] = sent_counts[c] / sent_counts["total"]

    sent_pct = sent_counts[
        base_group_cols
        + ["negative_share", "neutral_share", "positive_share"]
    ]

    event_period_stats = pd.merge(
        agg_basic, sent_pct, on=base_group_cols, how="left"
    )

    # 展平成每个事件一行
    pivot_cols = ["event_name", "event_date", "event_year"]
    ep_pivot = event_period_stats.pivot_table(
        index=pivot_cols,
        columns="period",
        values=[
            "n_posts",
            "mean_polarity",
            "negative_share",
            "neutral_share",
            "positive_share",
        ],
    )
    ep_pivot.columns = [
        f"{val}_{col}" for (val, col) in ep_pivot.columns.to_flat_index()
    ]
    ep_pivot = ep_pivot.reset_index()

    # 计算差值：post - baseline
    def safe_diff(col_post, col_base):
        return ep_pivot.get(col_post) - ep_pivot.get(col_base)

    ep_pivot["delta_n_posts"] = safe_diff("n_posts_post", "n_posts_baseline")
    ep_pivot["delta_mean_polarity"] = safe_diff(
        "mean_polarity_post", "mean_polarity_baseline"
    )

    print(f"[INFO] Computed event-level baseline vs post summary "
          f"for {len(ep_pivot)} events.")

    return ep_pivot


def summarize_event_trends(ep_summary: pd.DataFrame) -> pd.DataFrame:
    """Return year-level trend summary and also print it."""
    if ep_summary.empty:
        return pd.DataFrame()

    year_group = ep_summary.groupby("event_year").agg(
        avg_baseline_polarity=("mean_polarity_baseline", "mean"),
        avg_post_polarity=("mean_polarity_post", "mean"),
        avg_delta_polarity=("delta_mean_polarity", "mean"),
        avg_baseline_posts=("n_posts_baseline", "mean"),
        avg_delta_posts=("delta_n_posts", "mean"),
        n_events=("event_name", "count"),
    ).reset_index()

    print("\n[SUMMARY] Event-level trends by year:")
    print(year_group.round(4).to_string(index=False))

    return year_group


# ==========================================
# INSTANCE-LEVEL ANALYSIS
# ==========================================

def compute_instance_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute baseline vs post metrics at (event, instance) level,
    and compare to global averages for each event."""
    df_bp = df[df["period"].isin(["baseline", "post"])].copy()
    if df_bp.empty:
        print("[WARN] No data in baseline/post windows for instances.")
        return pd.DataFrame()

    group_cols = ["event_name", "event_date", "event_year", "instance", "period"]

    inst_basic = df_bp.groupby(group_cols)["polarity"].agg(
        n_posts="count",
        mean_polarity="mean"
    ).reset_index()

    global_group_cols = ["event_name", "event_date", "event_year", "period"]
    global_means = (
        df_bp.groupby(global_group_cols)["polarity"]
        .mean()
        .reset_index()
        .rename(columns={"polarity": "global_mean_polarity"})
    )

    inst_with_global = pd.merge(
        inst_basic,
        global_means,
        on=global_group_cols,
        how="left",
    )

    inst_with_global["diff_from_global"] = (
        inst_with_global["mean_polarity"] - inst_with_global["global_mean_polarity"]
    )

    pivot_cols = ["event_name", "event_date", "event_year", "instance"]
    inst_pivot = inst_with_global.pivot_table(
        index=pivot_cols,
        columns="period",
        values=["n_posts", "mean_polarity", "diff_from_global"],
    )
    inst_pivot.columns = [
        f"{val}_{col}" for (val, col) in inst_pivot.columns.to_flat_index()
    ]
    inst_pivot = inst_pivot.reset_index()

    def safe_diff(col_post, col_base):
        return inst_pivot.get(col_post) - inst_pivot.get(col_base)

    inst_pivot["delta_mean_polarity"] = safe_diff(
        "mean_polarity_post", "mean_polarity_baseline"
    )
    inst_pivot["delta_n_posts"] = safe_diff(
        "n_posts_post", "n_posts_baseline"
    )

    print(f"[INFO] Computed instance-level baseline vs post summary "
          f"for {len(inst_pivot)} (event, instance) combinations.")

    return inst_pivot


def compute_event_instance_heterogeneity(inst_summary: pd.DataFrame) -> pd.DataFrame:
    """Measure heterogeneity of instance-level sentiment for each event."""
    if inst_summary.empty:
        return pd.DataFrame()

    records = []

    for period_suffix in ["baseline", "post"]:
        col_name = f"mean_polarity_{period_suffix}"
        if col_name not in inst_summary.columns:
            continue

        sub = inst_summary.dropna(subset=[col_name])
        if sub.empty:
            continue

        grouped = sub.groupby(["event_name", "event_date", "event_year"])[col_name]
        stats = grouped.agg(
            mean_across_instances="mean",
            std_across_instances="std",
            n_instances="count",
        ).reset_index()
        stats["period"] = period_suffix
        records.append(stats)

    if not records:
        print("[WARN] No instance-level data for heterogeneity calculation.")
        return pd.DataFrame()

    hetero_df = pd.concat(records, ignore_index=True)
    print(f"[INFO] Computed heterogeneity stats for {len(hetero_df)} event-period rows.")
    return hetero_df


# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def plot_event_polarity_comparison(ep_summary: pd.DataFrame, filename: str):
    """每个事件：基线 vs 事后 平均情绪对比图"""
    if ep_summary.empty:
        return

    ep = ep_summary.sort_values("event_date").copy()
    labels = ep["event_name"].astype(str)
    x = range(len(ep))

    baseline = ep["mean_polarity_baseline"]
    post = ep["mean_polarity_post"]

    width = 0.4

    plt.figure(figsize=(max(8, len(ep) * 0.4), 5))
    ax = plt.gca()
    ax.bar([i - width/2 for i in x], baseline, width=width, label="Baseline (t0-7 ~ t0-1)")
    ax.bar([i + width/2 for i in x], post, width=width, label="Post (t0+1 ~ t0+7)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Sentiment Polarity")
    ax.set_title("Event-level Mean Sentiment: Baseline vs Post")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[INFO] Saved figure: {filename}")


def plot_event_volume_comparison(ep_summary: pd.DataFrame, filename: str):
    """每个事件：基线 vs 事后 帖子数量对比图"""
    if ep_summary.empty:
        return

    ep = ep_summary.sort_values("event_date").copy()
    labels = ep["event_name"].astype(str)
    x = range(len(ep))

    baseline = ep["n_posts_baseline"]
    post = ep["n_posts_post"]

    width = 0.4

    plt.figure(figsize=(max(8, len(ep) * 0.4), 5))
    ax = plt.gca()
    ax.bar([i - width/2 for i in x], baseline, width=width, label="Baseline (t0-7 ~ t0-1)")
    ax.bar([i + width/2 for i in x], post, width=width, label="Post (t0+1 ~ t0+7)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Number of Posts")
    ax.set_title("Event-level Volume: Baseline vs Post")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[INFO] Saved figure: {filename}")


def plot_year_trends_fig(year_summary: pd.DataFrame, filename: str):
    """按年份的平均基线/事后情绪 & 情绪变化、帖子变化"""
    if year_summary.empty:
        return

    ys = year_summary.sort_values("event_year").copy()
    years = ys["event_year"].astype(int)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # 上图：情绪
    ax1 = axes[0]
    ax1.plot(years, ys["avg_baseline_polarity"], marker="o", label="Baseline")
    ax1.plot(years, ys["avg_post_polarity"], marker="o", label="Post")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Mean Polarity")
    ax1.set_title("Yearly Average Sentiment (Baseline vs Post)")
    ax1.legend()

    # 下图：Δ 情绪 & 帖子
    ax2 = axes[1]
    ax2.plot(years, ys["avg_delta_polarity"], marker="o", label="Δ Mean Polarity (Post - Baseline)")
    ax2.set_ylabel("Δ Polarity")
    ax2.set_xlabel("Event Year")
    ax2.set_title("Yearly Average Change in Sentiment and Volume")

    # 双轴显示 Δ 帖子数量
    ax3 = ax2.twinx()
    ax3.plot(years, ys["avg_delta_posts"], marker="s", linestyle="--", label="Δ Posts (Post - Baseline)")
    ax3.set_ylabel("Δ Number of Posts")

    # 合并图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[INFO] Saved figure: {filename}")


def plot_instance_heterogeneity_fig(hetero_df: pd.DataFrame, filename: str):
    """每个事件：实例间情绪标准差（基线 vs 事后）"""
    if hetero_df.empty:
        return

    # 我们只用 std_across_instances
    # 展开成 baseline / post 两列
    pivot_cols = ["event_name", "event_date", "event_year"]
    het_pivot = hetero_df.pivot_table(
        index=pivot_cols,
        columns="period",
        values="std_across_instances"
    )
    het_pivot = het_pivot.reset_index().sort_values("event_date")

    labels = het_pivot["event_name"].astype(str)
    x = range(len(het_pivot))
    baseline = het_pivot.get("baseline")
    post = het_pivot.get("post")

    width = 0.4
    plt.figure(figsize=(max(8, len(het_pivot) * 0.4), 5))
    ax = plt.gca()
    if baseline is not None:
        ax.bar([i - width/2 for i in x], baseline, width=width, label="Baseline std across instances")
    if post is not None:
        ax.bar([i + width/2 for i in x], post, width=width, label="Post std across instances")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Std of Mean Polarity Across Instances")
    ax.set_title("Instance-level Sentiment Heterogeneity per Event")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[INFO] Saved figure: {filename}")


# ==========================================
# MAIN
# ==========================================

def main():
    df = load_data(DATA_FILE)
    if df.empty:
        return

    df = add_time_columns(df)

    # 事件层面分析
    event_summary = compute_event_level_summary(df)
    if not event_summary.empty:
        event_summary.to_csv(EVENT_SUMMARY_CSV, index=False)
        print(f"[INFO] Saved event-level summary to '{EVENT_SUMMARY_CSV}'.")

        year_summary = summarize_event_trends(event_summary)

        # 画图：事件级情绪 & 帖子
        plot_event_polarity_comparison(event_summary, FIG_EVENT_POLARITY)
        plot_event_volume_comparison(event_summary, FIG_EVENT_VOLUME)

        # 画图：年份趋势
        if not year_summary.empty:
            plot_year_trends_fig(year_summary, FIG_YEAR_TRENDS)

    # 实例层面分析
    inst_summary = compute_instance_level_summary(df)
    if not inst_summary.empty:
        inst_summary.to_csv(INSTANCE_SUMMARY_CSV, index=False)
        print(f"[INFO] Saved instance-level summary to '{INSTANCE_SUMMARY_CSV}'.")

        hetero_df = compute_event_instance_heterogeneity(inst_summary)
        if not hetero_df.empty:
            hetero_df.to_csv(HETEROGENEITY_CSV, index=False)
            print(f"[INFO] Saved event-instance heterogeneity stats to '{HETEROGENEITY_CSV}'.")
            plot_instance_heterogeneity_fig(hetero_df, FIG_INSTANCE_HETEROGENEITY)

    print("\n[DONE] Baseline vs post analysis with figures finished.")


if __name__ == "__main__":
    main()
