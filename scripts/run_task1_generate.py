"""
任务一主生成脚本：Monte Carlo 数据生成

用法：
    python scripts/run_task1_generate.py [--n-sims N] [--check-only]

参数：
    --n-sims    每个场景的模拟次数（默认 500）
    --check-only  仅运行自检，不重新生成数据（如已有数据）
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 无头模式
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 确保项目根在路径中
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulation.dgp import (
    N_FIRMS,
    YEARS,
    POST_YEAR,
    TAU,
    SCENARIOS,
    run_scenario,
    make_final_df,
)
from src.utils.paths import (
    ensure_dirs,
    raw_path,
    OUTPUT_CHECKS,
)


# ---------------------------------------------------------------------------
# 1. 数据生成
# ---------------------------------------------------------------------------

def generate_all_scenarios(n_sims: int = 500) -> dict[str, pd.DataFrame]:
    """
    生成全部场景的模拟数据，并保存到 data/raw/。

    Returns
    -------
    dict[str, pd.DataFrame]
        键为场景标签，值为对应场景的完整 DataFrame（调试版）
    """
    print(f"\n{'='*60}")
    print(f"  任务一：开始生成 {n_sims} 次模拟数据")
    print(f"{'='*60}\n")

    results: dict[str, pd.DataFrame] = {}

    for sc in SCENARIOS:
        t0 = time.time()
        print(f"[场景 {sc}] 开始 Monte Carlo 循环 ...")
        debug_df = run_scenario(sc, n_sims=n_sims, progress=True)

        # 保存正式交付版本（去除调试字段）
        final_df = make_final_df(debug_df)
        final_path = raw_path(sc, kind="final")
        final_df.to_parquet(final_path, index=False)
        print(f"  → 正式数据已保存：{final_path}")

        # 保存调试版本（保留 Y0/Y1/U，仅用于自检）
        debug_path = raw_path(sc, kind="debug")
        debug_df.to_parquet(debug_path, index=False)
        print(f"  → 调试数据已保存：{debug_path}")

        results[sc] = debug_df
        print(f"  场景 {sc} 完成，耗时 {time.time()-t0:.1f}s\n")

    return results


# ---------------------------------------------------------------------------
# 2. 自检
# ---------------------------------------------------------------------------

def treatment_rate_check(debug_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    自检 1：统计每个场景 500 次模拟的平均处理率。
    验收标准：500 次平均值落在 [0.28, 0.32] 区间内。
    """
    rows = []
    for sc, df in debug_dfs.items():
        # 每次 sim_id 的处理率
        rate_by_sim = df.groupby("sim_id")["D"].mean()
        rows.append({
            "场景": sc,
            "平均处理率": rate_by_sim.mean(),
            "标准差": rate_by_sim.std(),
            "最小": rate_by_sim.min(),
            "最大": rate_by_sim.max(),
            "验收": "✓" if 0.28 <= rate_by_sim.mean() <= 0.32 else "✗",
        })
    tbl = pd.DataFrame(rows)
    return tbl


def y0_trend_check(debug_dfs: dict[str, pd.DataFrame]) -> dict:
    """
    自检 2：绘制全部场景的 Y(0) 组均值趋势图。
    场景 A：处理组和控制组趋势应基本平行。
    场景 B：处理组趋势更陡，差异随时间累积。
    场景 C：同样趋势分歧，但由不可观测因素驱动。
    场景 D：沿用 B 的 Y(0) 机制，只额外引入处理效应异质性。
    """
    scenarios = list(SCENARIOS)
    n_sc = len(scenarios)

    # 上排展示两组均值路径，下排展示 treated - control 的 gap，更便于识别是否平行
    fig, axes = plt.subplots(
        2,
        n_sc,
        figsize=(5.2 * n_sc, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4]},
    )
    scenario_labels = {
        "A": "Scenario A\nY(0) Mean Trend",
        "B": "Scenario B\nY(0) Mean Trend",
        "C": "Scenario C\nY(0) Mean Trend",
        "D": "Scenario D\nY(0) Mean Trend",
    }

    for col, sc in enumerate(scenarios):
        ax_top = axes[0, col]
        ax_bottom = axes[1, col]
        df = debug_dfs[sc]
        mean_by_sim = (
            df.groupby(["sim_id", "D", "year"])["Y0"]
            .mean()
            .reset_index()
        )
        trend_map: dict[int, pd.Series] = {}

        for d_val, label, color, linestyle, marker in [
            (0, "Control", "steelblue", "--", "s"),
            (1, "Treatment", "coral", "-", "o"),
        ]:
            grp = mean_by_sim[mean_by_sim["D"] == d_val]
            trend = grp.groupby("year")["Y0"].mean()
            trend_map[d_val] = trend
            ax_top.plot(
                trend.index,
                trend.values,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=5,
                label=label,
                color=color,
                alpha=0.95,
            )

        gap = trend_map[1] - trend_map[0]
        ax_bottom.axhline(0.0, color="gray", linewidth=1, linestyle=":")
        ax_bottom.plot(
            gap.index,
            gap.values,
            color="black",
            linewidth=1.8,
            marker="D",
            markersize=4,
        )
        ax_bottom.fill_between(gap.index, 0.0, gap.values, color="gray", alpha=0.15)

        ax_top.set_title(f"{scenario_labels[sc]}\n(avg over simulations)", fontsize=11)
        ax_top.set_ylabel("Y(0) Mean")
        ax_top.legend(loc="best")
        ax_top.grid(alpha=0.3)

        ax_bottom.set_xlabel("Year")
        ax_bottom.set_ylabel("Gap")
        ax_bottom.grid(alpha=0.3)

    plt.tight_layout(pad=1.8, w_pad=2.0, h_pad=1.2)
    out_path = OUTPUT_CHECKS / "y0_trend_by_scenario.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return {"path": str(out_path)}


def propensity_score_check(debug_dfs: dict[str, pd.DataFrame]) -> dict:
    """
    自检 3：基于真实处理概率的分布对比图（共同支持检查）。
    如果处理组和控制组的 propensity score 分布重叠不足，说明共同支持可能严重缺失。
    """
    from scipy.special import expit

    scenarios = list(SCENARIOS)
    n_sc = len(scenarios)
    fig, axes = plt.subplots(1, n_sc, figsize=(5.2 * n_sc, 4.8), sharey=False)
    bins = np.linspace(0.0, 1.0, 31)

    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, scenarios):
        df = debug_dfs[sc]
        firm_level = df[df["year"] == 1].copy()

        if sc == "A":
            p = np.full(len(firm_level), 0.30)
        elif sc in {"B", "D"}:
            p = expit(-0.95 + 0.8 * firm_level["X"].values)
        else:
            p = expit(-0.95 + 0.2 * firm_level["X"].values + 0.8 * firm_level["U"].values)

        firm_level["propensity"] = p

        for d_val, label, color in [(0, "Control", "steelblue"), (1, "Treatment", "coral")]:
            grp = firm_level[firm_level["D"] == d_val]["propensity"]
            ax.hist(
                grp,
                bins=bins,
                alpha=0.35,
                label=label,
                color=color,
                density=True,
                edgecolor=color,
                linewidth=1.2,
            )

        ax.set_title(f"Scenario {sc}: Propensity Score Distribution")
        ax.set_xlabel("Propensity Score")
        ax.set_ylabel("Density")
        ax.set_xlim(0.0, 1.0)
        ax.legend()
        ax.grid(alpha=0.3)

        if sc == "A":
            ax.axvline(0.30, color="black", linestyle="--", linewidth=1.3)
            ax.text(
                0.04,
                0.95,
                "Random assignment:\nall firms have true p = 0.30",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
            print("  Scenario A common support: fixed at 0.300 (random assignment)")
        else:
            # 打印共同支持重叠情况
            treated_ps = firm_level[firm_level["D"] == 1]["propensity"]
            control_ps = firm_level[firm_level["D"] == 0]["propensity"]
            overlap_min = max(treated_ps.min(), control_ps.min())
            overlap_max = min(treated_ps.max(), control_ps.max())
            ax.axvspan(overlap_min, overlap_max, color="darkgreen", alpha=0.10)
            ax.text(
                0.04,
                0.95,
                f"Overlap:\n[{overlap_min:.2f}, {overlap_max:.2f}]",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
            print(f"  Scenario {sc} common support: [{overlap_min:.3f}, {overlap_max:.3f}]")

    plt.tight_layout(pad=1.6, w_pad=2.0)
    out_path = OUTPUT_CHECKS / "propensity_score_check.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return {"path": str(out_path)}


def generate_checks_report(
    debug_dfs: dict[str, pd.DataFrame],
    rate_tbl: pd.DataFrame,
    n_sims: int,
) -> Path:
    """生成 Markdown 格式自检报告（不依赖 tabulate）"""
    report_path = OUTPUT_CHECKS / "task1_self_check_report.md"

    # 手写 Markdown 表格
    headers = ["场景", "平均处理率", "标准差", "最小", "最大", "验收"]
    rows = [headers]
    for _, row in rate_tbl.iterrows():
        rows.append([str(row[c]) for c in headers])
    col_widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    table_lines = []
    for ri, row in enumerate(rows):
        table_lines.append("| " + " | ".join(v.ljust(col_widths[j]) for j, v in enumerate(row)) + " |")
        if ri == 0:
            table_lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")

    lines = [
        "# 任务一自检报告",
        "",
        f"生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"模拟次数：{N_FIRMS} 企业 × 6 年 × {n_sims} 次",
        f"处理效应说明：场景 A/B/C 为常数 τ = {TAU}",
        "场景 D 为异质处理效应：τ_i = 50 + 8 * clip(X_i, -1.5, 1.5)",
        "",
        "## 1. 处理组占比检查",
        "",
        *table_lines,
        "",
        f"> 验收标准：{n_sims} 次平均处理率落在 [0.28, 0.32] 区间",
        "",
        "## 2. Y(0) 组均值趋势图",
        "",
        "![Y(0) 趋势图](y0_trend_by_scenario.png)",
        "",
        "## 3. 倾向得分分布（全部场景）",
        "",
        "![倾向得分分布](propensity_score_check.png)",
        "",
        "## 4. 随机种子与复现说明",
        "",
        "- 场景 A：`seed = 1000 + sim_id`",
        "- 场景 B：`seed = 2000 + sim_id`",
        "- 场景 C：`seed = 3000 + sim_id`",
        "- 场景 D：`seed = 4000 + sim_id`",
        "- 每次循环内部创建独立 RNG，不依赖全局状态",
        "",
        "## 5. 正式交付数据字段",
        "",
        "| 字段 | 说明 |",
        "| :--- | :--- |",
        "| scenario | 场景标签（A / B / C / D） |",
        f"| sim_id | 模拟编号 1-{n_sims} |",
        "| firm_id | 企业编号 0-999 |",
        "| year | 年份 1-6 |",
        "| post | 处理后指示变量（t≥4 时为 1） |",
        "| D | 处理分配（1=处理组，0=控制组） |",
        "| X | 处理前协变量（企业初始规模 ~ N(0,1)） |",
        "| Y | 观测结果（万元） |",
        "",
        "正式交付数据**不含** U_i、Y0、Y1 等调试字段。",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="任务一：Monte Carlo 数据生成")
    parser.add_argument("--n-sims", type=int, default=500)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    print(f"\n任务一主脚本 — Python {sys.version.split()[0]}")
    print(f"每个场景模拟次数：{args.n_sims}")

    if args.check_only:
        print("\n[check-only 模式] 跳过数据生成，尝试加载已有数据 ...")
        debug_dfs = {}
        for sc in SCENARIOS:
            p = raw_path(sc, kind="debug")
            if p.exists():
                debug_dfs[sc] = pd.read_parquet(p)
                print(f"  场景 {sc}: 已加载 {p}")
            else:
                print(f"  场景 {sc}: 未找到数据，退出。")
                sys.exit(1)
    else:
        debug_dfs = generate_all_scenarios(n_sims=args.n_sims)

    # ---- 自检 ----
    print("\n开始自检 ...")
    rate_tbl = treatment_rate_check(debug_dfs)
    print("\nTreatment rate check:")
    print(rate_tbl.to_string(index=False))

    y0_trend_check(debug_dfs)
    print("\nY(0) 趋势图已保存到 outputs/checks/y0_trend_by_scenario.png")

    propensity_score_check(debug_dfs)
    print("\n倾向得分分布图已保存到 outputs/checks/propensity_score_check.png")

    report_path = generate_checks_report(debug_dfs, rate_tbl, n_sims=args.n_sims)
    print(f"\n自检报告已保存：{report_path}")

    print(f"\n{'='*60}")
    print("  任务一数据生成完成！")
    print(f"{'='*60}")
    print("  下一步可运行：python scripts/run_task2_estimate.py")


if __name__ == "__main__":
    main()
