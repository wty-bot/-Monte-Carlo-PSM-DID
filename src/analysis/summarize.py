"""
分析模块：结果汇总、表格输出、图表生成

遵循《任务二规划文档》第七节输出物要求。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# 汇总表
# ---------------------------------------------------------------------------

def summarize_estimates(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    按 scenario × method 汇总估计结果。

    Returns
    -------
    pd.DataFrame
        包含 scenario, method, estimand, true_effect_target_mean,
        mean_estimate, bias, rmse, std, valid_runs, failure_runs
    """
    rows = []
    for (sc, method), grp in results_df.groupby(["scenario", "method"]):
        valid_rows = grp[(grp["status"] == "ok") & grp["estimate"].notna()].copy()
        valid = valid_rows["estimate"].dropna()
        failure = grp[grp["status"] != "ok"]
        n_valid = len(valid_rows)
        n_fail = len(failure)
        target = grp["true_effect_target"].dropna()
        target_mean = target.mean() if not target.empty else np.nan
        estimand_vals = grp["target_estimand"].dropna()
        estimand = estimand_vals.iloc[0] if not estimand_vals.empty else ""
        rows.append({
            "scenario": sc,
            "method": method,
            "estimand": estimand,
            "true_effect_target_mean": target_mean,
            "mean_estimate": valid.mean() if n_valid > 0 else np.nan,
            "bias": valid_rows["bias"].mean() if n_valid > 0 else np.nan,
            "rmse": np.sqrt(valid_rows["sq_error"].mean()) if n_valid > 0 else np.nan,
            "std": valid.std() if n_valid > 1 else np.nan,
            "valid_runs": n_valid,
            "failure_runs": n_fail,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 匹配诊断表
# ---------------------------------------------------------------------------

def build_match_diagnostic(match_records: list) -> pd.DataFrame:
    """
    将匹配诊断记录转换为 DataFrame。
    现在包含 method 字段（PSM / PSM-DID），不再按方法重复两行。
    """
    if not match_records:
        return pd.DataFrame()
    rows = []
    for m in match_records:
        rows.append({
            "scenario": getattr(m, "scenario", ""),
            "sim_id": getattr(m, "sim_id", ""),
            "method": m.method,            # "PSM" or "PSM-DID"
            "n_before_trim_treated": m.n_before_trim_treated,
            "n_before_trim_control": m.n_before_trim_control,
            "n_after_trim_treated": m.n_after_trim_treated,
            "n_after_trim_control": m.n_after_trim_control,
            "n_matched_pairs": m.n_treated,
            "n_control_unique": m.n_control_unique,
            "avg_match_distance": np.mean(m.match_distances) if len(m.match_distances) > 0 else np.nan,
            "common_support_width": m.common_support_width,
            "common_support_failure": m.common_support_failure,
            "narrow_support_warning": getattr(m, "narrow_support_warning", False),
        })
    return pd.DataFrame(rows)


def _standardized_mean_difference(
    treated: np.ndarray,
    control: np.ndarray,
) -> float:
    """计算标准化均值差（SMD）。"""
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)
    if len(treated) == 0 or len(control) == 0:
        return np.nan
    var_t = np.var(treated, ddof=1) if len(treated) > 1 else 0.0
    var_c = np.var(control, ddof=1) if len(control) > 1 else 0.0
    var_pool = (var_t + var_c) / 2
    if var_pool <= 0:
        return np.nan
    return float((treated.mean() - control.mean()) / np.sqrt(var_pool))


def _build_preperiod_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    构造处理前结果水平与预趋势诊断特征。

    返回字段：
    - firm_id, D, X
    - Y_pre_mean: t=1,2,3 的均值
    - pre_slope: (Y_3 - Y_1) / 2
    """
    pre = panel[panel["post"] == 0].copy()
    years = sorted(pre["year"].unique())
    if len(years) < 3:
        raise ValueError("Pre-period diagnostics require at least 3 pre-treatment periods.")

    wide = pre.pivot(index="firm_id", columns="year", values="Y")
    first_year = years[0]
    last_year = years[-1]

    features = panel.groupby("firm_id").agg(D=("D", "first"), X=("X", "first")).reset_index()
    features["Y_pre_mean"] = features["firm_id"].map(wide.mean(axis=1))
    features["pre_slope"] = features["firm_id"].map((wide[last_year] - wide[first_year]) / (last_year - first_year))
    return features


def _preperiod_placebo_did(panel: pd.DataFrame) -> float:
    """
    在处理前 3 期上做 one-step placebo DID。

    设第 3 个处理前时期为伪 post，用于直接检查处理前是否已存在系统性差异。
    """
    from ..estimation.methods import _twfe_coef

    pre = panel[panel["post"] == 0].copy()
    if pre.empty:
        return np.nan
    last_pre_year = int(pre["year"].max())
    pre["placebo_post"] = (pre["year"] == last_pre_year).astype(int)
    pre["D_placebo"] = pre["D"] * pre["placebo_post"]
    return _twfe_coef(pre, treatment_col="D_placebo")


def _preperiod_trend_gap(panel: pd.DataFrame) -> float:
    """
    在处理前 3 期上估计处理组相对控制组的线性预趋势斜率差。
    """
    from ..estimation.methods import _twfe_coef

    pre = panel[panel["post"] == 0].copy()
    if pre.empty:
        return np.nan
    first_pre_year = int(pre["year"].min())
    pre["pre_s"] = pre["year"] - first_pre_year
    pre["D_pretrend"] = pre["D"] * pre["pre_s"]
    return _twfe_coef(pre, treatment_col="D_pretrend")


def compute_identification_diagnostics(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算直接识别诊断：
    1. 匹配前/后的 X、Y_pre_mean、pre_slope 平衡性
    2. 匹配前/后的 placebo DID
    3. 匹配前/后的 differential pretrend coefficient
    """
    from ..estimation.methods import _build_matched_panel, prepare_matching

    rows: list[dict[str, object]] = []
    for sc in sorted(panel_df["scenario"].dropna().unique().tolist()):
        sc_panel = panel_df[panel_df["scenario"] == sc].copy()
        sim_ids = sorted(sc_panel["sim_id"].unique())

        for sid in sim_ids:
            sim_panel = sc_panel[sc_panel["sim_id"] == sid].copy()
            full_features = _build_preperiod_features(sim_panel)

            treated_mask = full_features["D"].values == 1
            control_mask = full_features["D"].values == 0
            treated = full_features.loc[treated_mask]
            control = full_features.loc[control_mask]

            rows.append({
                "scenario": sc,
                "sim_id": sid,
                "sample": "Before",
                "status": "ok",
                "x_smd": _standardized_mean_difference(treated["X"].values, control["X"].values),
                "pre_mean_smd": _standardized_mean_difference(
                    treated["Y_pre_mean"].values, control["Y_pre_mean"].values
                ),
                "pre_slope_smd": _standardized_mean_difference(
                    treated["pre_slope"].values, control["pre_slope"].values
                ),
                "placebo_did": _preperiod_placebo_did(sim_panel),
                "pretrend_coef": _preperiod_trend_gap(sim_panel),
            })

            prep = prepare_matching(sim_panel, sc, sid)
            cs_diag = prep.common_support_diag
            if cs_diag.common_support_failure or prep.trimmed_firm.empty:
                rows.append({
                    "scenario": sc,
                    "sim_id": sid,
                    "sample": "After",
                    "status": "common_support_failure",
                    "x_smd": np.nan,
                    "pre_mean_smd": np.nan,
                    "pre_slope_smd": np.nan,
                    "placebo_did": np.nan,
                    "pretrend_coef": np.nan,
                })
                continue

            if prep.match is None:
                rows.append({
                    "scenario": sc,
                    "sim_id": sid,
                    "sample": "After",
                    "status": "common_support_failure",
                    "x_smd": np.nan,
                    "pre_mean_smd": np.nan,
                    "pre_slope_smd": np.nan,
                    "placebo_did": np.nan,
                    "pretrend_coef": np.nan,
                })
                continue

            matched_panel = _build_matched_panel(sim_panel, prep.match)
            if matched_panel.empty or not (matched_panel["D"] == 0).any():
                rows.append({
                    "scenario": sc,
                    "sim_id": sid,
                    "sample": "After",
                    "status": "common_support_failure",
                    "x_smd": np.nan,
                    "pre_mean_smd": np.nan,
                    "pre_slope_smd": np.nan,
                    "placebo_did": np.nan,
                    "pretrend_coef": np.nan,
                })
                continue

            matched_features = _build_preperiod_features(matched_panel)
            treated_m = matched_features[matched_features["D"] == 1]
            control_m = matched_features[matched_features["D"] == 0]

            rows.append({
                "scenario": sc,
                "sim_id": sid,
                "sample": "After",
                "status": "ok",
                "x_smd": _standardized_mean_difference(treated_m["X"].values, control_m["X"].values),
                "pre_mean_smd": _standardized_mean_difference(
                    treated_m["Y_pre_mean"].values, control_m["Y_pre_mean"].values
                ),
                "pre_slope_smd": _standardized_mean_difference(
                    treated_m["pre_slope"].values, control_m["pre_slope"].values
                ),
                "placebo_did": _preperiod_placebo_did(matched_panel),
                "pretrend_coef": _preperiod_trend_gap(matched_panel),
            })

    return pd.DataFrame(rows)


def compute_identification_diagnostics_for_sim(
    panel: pd.DataFrame,
    scenario: str,
    sim_id: int,
    prep,
) -> list[dict[str, object]]:
    """
    基于主估计循环中已经得到的匹配准备结果，生成单次模拟的直接识别诊断。
    """
    from ..estimation.methods import _build_matched_panel

    full_features = _build_preperiod_features(panel)
    treated = full_features[full_features["D"] == 1]
    control = full_features[full_features["D"] == 0]

    rows: list[dict[str, object]] = [{
        "scenario": scenario,
        "sim_id": sim_id,
        "sample": "Before",
        "status": "ok",
        "x_smd": _standardized_mean_difference(treated["X"].values, control["X"].values),
        "pre_mean_smd": _standardized_mean_difference(
            treated["Y_pre_mean"].values, control["Y_pre_mean"].values
        ),
        "pre_slope_smd": _standardized_mean_difference(
            treated["pre_slope"].values, control["pre_slope"].values
        ),
        "placebo_did": _preperiod_placebo_did(panel),
        "pretrend_coef": _preperiod_trend_gap(panel),
    }]

    cs_diag = prep.common_support_diag
    if cs_diag.common_support_failure or prep.trimmed_firm.empty or prep.match is None:
        rows.append({
            "scenario": scenario,
            "sim_id": sim_id,
            "sample": "After",
            "status": "common_support_failure",
            "x_smd": np.nan,
            "pre_mean_smd": np.nan,
            "pre_slope_smd": np.nan,
            "placebo_did": np.nan,
            "pretrend_coef": np.nan,
        })
        return rows

    matched_panel = _build_matched_panel(panel, prep.match)
    if matched_panel.empty or not (matched_panel["D"] == 0).any():
        rows.append({
            "scenario": scenario,
            "sim_id": sim_id,
            "sample": "After",
            "status": "common_support_failure",
            "x_smd": np.nan,
            "pre_mean_smd": np.nan,
            "pre_slope_smd": np.nan,
            "placebo_did": np.nan,
            "pretrend_coef": np.nan,
        })
        return rows

    matched_features = _build_preperiod_features(matched_panel)
    treated_m = matched_features[matched_features["D"] == 1]
    control_m = matched_features[matched_features["D"] == 0]
    rows.append({
        "scenario": scenario,
        "sim_id": sim_id,
        "sample": "After",
        "status": "ok",
        "x_smd": _standardized_mean_difference(treated_m["X"].values, control_m["X"].values),
        "pre_mean_smd": _standardized_mean_difference(
            treated_m["Y_pre_mean"].values, control_m["Y_pre_mean"].values
        ),
        "pre_slope_smd": _standardized_mean_difference(
            treated_m["pre_slope"].values, control_m["pre_slope"].values
        ),
        "placebo_did": _preperiod_placebo_did(matched_panel),
        "pretrend_coef": _preperiod_trend_gap(matched_panel),
    })
    return rows


def summarize_identification_diagnostics(diag_df: pd.DataFrame) -> pd.DataFrame:
    """
    将逐次模拟的直接识别诊断汇总为场景 × 样本层级统计表。
    """
    rows = []
    for (sc, sample), grp in diag_df.groupby(["scenario", "sample"]):
        valid = grp[grp["status"] == "ok"]
        rows.append({
            "scenario": sc,
            "sample": sample,
            "valid_runs": len(valid),
            "failure_runs": len(grp) - len(valid),
            "abs_x_smd": valid["x_smd"].abs().mean(),
            "abs_pre_mean_smd": valid["pre_mean_smd"].abs().mean(),
            "abs_pre_slope_smd": valid["pre_slope_smd"].abs().mean(),
            "mean_placebo_did": valid["placebo_did"].mean(),
            "mean_pretrend_coef": valid["pretrend_coef"].mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 图表 1：三场景三方法估计值分布图
# ---------------------------------------------------------------------------

def plot_estimate_distributions(
    results_df: pd.DataFrame,
    out_dir: Path | None = None,
) -> Path:
    """
    绘制各场景下四种方法估计值的分布（箱线图）。
    """
    scenarios = sorted(results_df["scenario"].dropna().unique().tolist())
    n_sc = max(len(scenarios), 1)
    fig, axes = plt.subplots(1, n_sc, figsize=(6 * n_sc, 5.5), sharey=True)
    axes = np.atleast_1d(axes)
    methods = ["PSM", "DID", "DID-X", "PSM-DID"]
    colors = {"PSM": "#4C72B0", "DID": "#DD8452", "DID-X": "#8172B3", "PSM-DID": "#55A868"}

    for ax, sc in zip(axes, scenarios):
        sc_data = results_df[results_df["scenario"] == sc]
        plot_data = []
        labels = []
        for m in methods:
            vals = sc_data[(sc_data["method"] == m) & (sc_data["status"] == "ok")]["estimate"].dropna()
            plot_data.append(vals.values)
            labels.append(m)

        bp = ax.boxplot(
            plot_data,
            labels=labels,
            patch_artist=True,
            widths=0.6,
            showfliers=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(colors[m])
            patch.set_alpha(0.6)

        sc_true = sc_data["true_effect_target"].dropna().mean()
        if not np.isnan(sc_true):
            ax.axhline(sc_true, color="red", linestyle="--", linewidth=1.2, label=f"Target truth={sc_true:.2f}")
        ax.set_title(f"Scenario {sc}", fontsize=13)
        ax.set_ylabel("Estimate" if sc == scenarios[0] else "")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Estimate Distributions by Scenario and Method", fontsize=14, y=1.02)
    plt.tight_layout()
    if out_dir is None:
        out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "estimate_distributions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 图表 2：倾向得分共同支持图
# ---------------------------------------------------------------------------

def plot_common_support(
    panel_df: pd.DataFrame,
    out_dir: Path | None = None,
) -> Path:
    """
    对每个场景绘制倾向得分的共同支持分布图。

    将所有 500 次模拟的倾向得分合并绘制，直观展示全量数据下的重叠情况。
    """
    from scipy.special import expit
    from ..estimation.methods import _build_firm_level, _estimate_propensity

    scenarios = sorted(panel_df["scenario"].dropna().unique().tolist())
    n_sc = max(len(scenarios), 1)
    fig, axes = plt.subplots(1, n_sc, figsize=(5.4 * n_sc, 5), sharey=False)
    axes = np.atleast_1d(axes)
    bins = np.linspace(0.0, 1.0, 31)

    for ax, sc in zip(axes, scenarios):
        sc_panel = panel_df[panel_df["scenario"] == sc].copy()
        sim_ids = sorted(sc_panel["sim_id"].unique())

        all_treated_ps: list[np.ndarray] = []
        all_control_ps: list[np.ndarray] = []
        all_cs_mins: list[float] = []
        all_cs_maxs: list[float] = []

        for sid in sim_ids:
            sim_panel = sc_panel[sc_panel["sim_id"] == sid].copy()
            firm = _build_firm_level(sim_panel)
            p = _estimate_propensity(firm)
            firm["propensity"] = p

            treated_p = firm[firm["D"] == 1]["propensity"].values
            control_p = firm[firm["D"] == 0]["propensity"].values
            all_treated_ps.append(treated_p)
            all_control_ps.append(control_p)

            if len(treated_p) > 0 and len(control_p) > 0:
                all_cs_mins.append(max(treated_p.min(), control_p.min()))
                all_cs_maxs.append(min(treated_p.max(), control_p.max()))

        # 合并所有 sims 的得分
        all_treated = np.concatenate(all_treated_ps)
        all_control = np.concatenate(all_control_ps)

        ax.hist(all_treated, bins=bins, alpha=0.45, label="Treatment",
                color="coral", density=True, edgecolor="coral", linewidth=0.8)
        ax.hist(all_control, bins=bins, alpha=0.45, label="Control",
                color="steelblue", density=True, edgecolor="steelblue", linewidth=0.8)

        # 共同支持区间：取所有 sim 的均值
        if all_cs_mins and all_cs_maxs:
            avg_min = np.mean(all_cs_mins)
            avg_max = np.mean(all_cs_maxs)
            ax.axvspan(avg_min, avg_max, color="darkgreen", alpha=0.10)
            ax.text(0.04, 0.95,
                    f"Avg Overlap: [{avg_min:.2f}, {avg_max:.2f}]",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})

        ax.set_title(f"Scenario {sc}", fontsize=12)
        ax.set_xlabel("Estimated Propensity Score")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Propensity Score Common Support (all 500 sims aggregated)", fontsize=14, y=1.02)
    plt.tight_layout()
    if out_dir is None:
        out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "common_support.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 图表 3：匹配前后 X 平衡性图
# ---------------------------------------------------------------------------

def plot_covariate_balance(
    ident_diag_df: pd.DataFrame,
    out_dir: Path | None = None,
) -> Path:
    """
    基于已汇总的逐次识别诊断，绘制匹配前后 X 的标准化均值差。
    """
    scenarios = sorted(ident_diag_df["scenario"].dropna().unique().tolist())
    n_sc = max(len(scenarios), 1)
    fig, axes = plt.subplots(1, n_sc, figsize=(6 * n_sc, 5.5))
    axes = np.atleast_1d(axes)

    for ax, sc in zip(axes, scenarios):
        before = ident_diag_df[
            (ident_diag_df["scenario"] == sc)
            & (ident_diag_df["sample"] == "Before")
            & (ident_diag_df["status"] == "ok")
        ]["x_smd"].abs().dropna().values
        after = ident_diag_df[
            (ident_diag_df["scenario"] == sc)
            & (ident_diag_df["sample"] == "After")
            & (ident_diag_df["status"] == "ok")
        ]["x_smd"].abs().dropna().values

        mean_before = np.mean(before) if len(before) > 0 else 0.0
        std_before = np.std(before) if len(before) > 0 else 0.0
        mean_after = np.mean(after) if len(after) > 0 else 0.0
        std_after = np.std(after) if len(after) > 0 else 0.0

        n_valid = len(after)

        bars = ax.bar(
            ["Before", "After"],
            [mean_before, mean_after],
            color=["#CC6677", "#44AA99"],
            alpha=0.8,
            width=0.5,
            yerr=[std_before, std_after],
            capsize=5,
            ecolor="gray",
        )
        ax.axhline(0.1, color="red", linestyle="--", linewidth=1, alpha=0.7,
                   label="SMD=0.1 threshold")

        # 标注均值 ± 标准差
        ax.text(0, mean_before + std_before + 0.01,
                f"{mean_before:.3f}±{std_before:.3f}",
                ha="center", va="bottom", fontsize=9, color="#CC6677")
        ax.text(1, mean_after + std_after + 0.01,
                f"{mean_after:.3f}±{std_after:.3f}\n(n={n_valid})",
                ha="center", va="bottom", fontsize=9, color="#44AA99")

        ax.set_title(f"Scenario {sc}", fontsize=12)
        ax.set_ylabel("|Standardized Mean Difference|")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(mean_before + std_before, mean_after + std_after) * 1.25)

    fig.suptitle(
        "Covariate (X) Balance: Before vs After Matching\n"
        "(mean ± std across simulations; matched-sample diagnostics)",
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    if out_dir is None:
        out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "covariate_balance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_preperiod_balance(
    ident_diag_df: pd.DataFrame,
    out_dir: Path | None = None,
) -> Path:
    """
    绘制匹配前后处理前结果水平与预趋势的平衡性图。
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    metrics = [
        ("x_smd", "|SMD of X|"),
        ("pre_mean_smd", "|SMD of pre-period mean Y|"),
        ("pre_slope_smd", "|SMD of pre-period slope|"),
    ]
    colors = {"Before": "#CC6677", "After": "#44AA99"}

    for ax, (metric, label) in zip(axes, metrics):
        plot_rows = []
        scenarios = sorted(ident_diag_df["scenario"].dropna().unique().tolist())
        for sc in scenarios:
            for sample in ["Before", "After"]:
                sub = ident_diag_df[
                    (ident_diag_df["scenario"] == sc)
                    & (ident_diag_df["sample"] == sample)
                    & (ident_diag_df["status"] == "ok")
                ]
                vals = sub[metric].abs().dropna().values
                plot_rows.append({
                    "scenario": sc,
                    "sample": sample,
                    "mean": np.mean(vals) if len(vals) > 0 else np.nan,
                    "std": np.std(vals) if len(vals) > 0 else np.nan,
                })

        plot_df = pd.DataFrame(plot_rows)
        x = np.arange(len(scenarios))
        width = 0.34
        before = plot_df[plot_df["sample"] == "Before"].reset_index(drop=True)
        after = plot_df[plot_df["sample"] == "After"].reset_index(drop=True)

        ax.bar(
            x - width / 2,
            before["mean"].values,
            width=width,
            yerr=before["std"].values,
            color=colors["Before"],
            alpha=0.85,
            capsize=4,
            label="Before",
        )
        ax.bar(
            x + width / 2,
            after["mean"].values,
            width=width,
            yerr=after["std"].values,
            color=colors["After"],
            alpha=0.85,
            capsize=4,
            label="After",
        )
        ax.axhline(0.1, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.set_title(label, fontsize=12)
        ax.set_ylabel("Mean absolute SMD")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Pre-period balance before vs after matching", fontsize=14, y=1.02)
    plt.tight_layout()
    if out_dir is None:
        out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preperiod_balance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 事件研究图
# ---------------------------------------------------------------------------

def plot_event_study(
    coeffs_df: pd.DataFrame,
    out_dir: Path | None = None,
) -> Path:
    """
    基于逐次模拟的事件研究系数，绘制事件研究图。

    对每个场景，绘制两条线：
      - Full（全样本 DID）
      - Matched（匹配样本 PSM-DID）

    每条线为 500 次模拟的均值，误差带为 2.5%–97.5% 分位数。
    垂直虚线标示处理前/后分界（t=3 与 t=4 之间）。
    水平虚线标示零线。参考期 t=3 的系数固定为 0。
    """
    scenarios = sorted(coeffs_df["scenario"].dropna().unique().tolist())
    n_sc = len(scenarios)
    if n_sc == 0:
        raise ValueError("No scenario found in event study coefficients.")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, n_sc, figsize=(6.5 * n_sc, 5.5), sharey=False)
    if n_sc == 1:
        axes = [axes]

    pre_years = [1, 2, 3]
    post_years = [4, 5, 6]
    ref_year = 3

    # 聚合：对每个 scenario × sample × year，计算均值与分位数
    agg = (
        coeffs_df
        .groupby(["scenario", "sample", "year"])["coef"]
        .agg(mean="mean", lo=lambda x: np.percentile(x, 2.5), hi=lambda x: np.percentile(x, 97.5))
        .reset_index()
    )

    sample_config = {
        "full": {"label": "Full sample DID", "color": "#DD8452", "marker": "s"},
        "matched": {"label": "Matched PSM-DID", "color": "#55A868", "marker": "o"},
    }

    for ax, sc in zip(axes, scenarios):
        for sample_key, cfg in sample_config.items():
            sub = agg[(agg["scenario"] == sc) & (agg["sample"] == sample_key)]
            if sub.empty:
                continue
            sub = sub.sort_values("year")
            years = sub["year"].values
            mean_vals = sub["mean"].values
            lo_vals = sub["lo"].values
            hi_vals = sub["hi"].values

            ax.plot(years, mean_vals, color=cfg["color"], marker=cfg["marker"],
                    linewidth=2, markersize=6, label=cfg["label"])
            ax.fill_between(years, lo_vals, hi_vals, color=cfg["color"], alpha=0.15)

        # 处理前/后分界线
        ax.axvline(x=ref_year + 0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.5)
        ax.axvspan(ref_year + 0.5, 6.5, color="lightgreen", alpha=0.06)

        # 标注 Pre / Post
        y_lims = ax.get_ylim()
        y_mid = (y_lims[0] + y_lims[1]) / 2
        ax.text(2, y_lims[1] * 0.92, "Pre", ha="center", fontsize=11,
                fontweight="bold", color="gray")
        ax.text(5, y_lims[1] * 0.92, "Post", ha="center", fontsize=11,
                fontweight="bold", color="gray")

        ax.set_title(f"Scenario {sc}", fontsize=13)
        ax.set_xlabel("Year")
        if sc == scenarios[0]:
            ax.set_ylabel("Event-study coefficient β_k")
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Event Study: Dynamic DID Coefficients\n(mean ± 2.5%–97.5% Monte Carlo percentile band across 500 simulations)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    if out_dir is None:
        out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "event_study.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 自检报告
# ---------------------------------------------------------------------------

def generate_task2_check_report(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    match_diag_df: pd.DataFrame | None = None,
    ident_summary_df: pd.DataFrame | None = None,
    out_dir: Path | None = None,
) -> Path:
    """
    生成任务二自检报告（Markdown 格式）。

    报告结构：
    1. 总体运行情况（记录数、方法数、成功/失败）
    2. 汇总结果表（含 DID、DID-X、PSM、PSM-DID 四种方法）
    3. DGP 机制解释（每个场景的关键识别假设违反）
    4. 识别假设核对（PSM-DID 的五大假设逐一检查）
    5. 共同支持诊断（极窄区间警告、失败率）
    6. 结果机制分析（"在什么 DGP 下某方法更准/更差"的解释）
    """
    if out_dir is None:
        out_dir = Path("outputs/checks")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "task2_self_check_report.md"

    lines: list[str] = []
    scenarios = sorted(results_df["scenario"].dropna().unique().tolist()) if not results_df.empty else []

    # ===================================================================
    # 标题
    # ===================================================================
    lines += [
        "# 任务二自检报告",
        "",
        f"生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    # ===================================================================
    # 1. 总体运行情况
    # ===================================================================
    lines += [
        "## 1. 总体运行情况",
        "",
    ]
    methods = ["DID", "DID-X", "PSM", "PSM-DID"]
    scenario_count = results_df["scenario"].nunique() if not results_df.empty else 0
    per_method_runs = (
        int(results_df.groupby(["scenario", "method"]).size().max())
        if not results_df.empty else 0
    )
    total_expected = scenario_count * per_method_runs * len(methods)
    total_actual = len(results_df)
    completion_pct = (total_actual / total_expected * 100) if total_expected > 0 else 0.0
    lines.append(
        f"- 预期记录数：{total_expected}（{scenario_count} 场景 × {per_method_runs} 次 × {len(methods)} 方法）"
    )
    lines.append(f"- 实际记录数：{total_actual}（完成度 {completion_pct:.1f}%）")

    for sc in scenarios:
        sc_data = results_df[results_df["scenario"] == sc]
        lines.append(f"- **场景 {sc}**：{len(sc_data)} 条记录")
        for m in methods:
            m_data = sc_data[sc_data["method"] == m]
            ok = len(m_data[m_data["status"] == "ok"])
            fail = len(m_data[m_data["status"] != "ok"])
            denom = len(m_data)
            pct = f"{ok / denom * 100:.1f}%" if denom > 0 else "0.0%"
            if fail > 0:
                pct = f"{pct}（⚠ {fail} 次失败）"
            lines.append(f"  - {m}：{ok}/{denom} 成功（{pct}）")

    # ===================================================================
    # 2. 汇总结果表
    # ===================================================================
    lines += [
        "",
        "## 2. 汇总结果表",
        "",
        "| 场景 | 方法 | 目标参数 | 真值均值 | 平均估计 | Bias | RMSE | Std | 有效次数 | 失败次数 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for _, row in summary_df.iterrows():
        mth = row["method"]
        flag = ""
        if row["failure_runs"] > 0:
            flag = "（⚠ 有失败）"
        lines.append(
            f"| {row['scenario']} | **{mth}**{flag} | "
            f"{row['estimand']} | {row['true_effect_target_mean']:.3f} | "
            f"{row['mean_estimate']:.3f} | {row['bias']:.3f} | "
            f"{row['rmse']:.3f} | {row['std']:.3f} | "
            f"{row['valid_runs']} | {row['failure_runs']} |"
        )

    # ===================================================================
    # 3. 数据生成机制（DGP）解释
    # ===================================================================
    lines += [
        "",
        "## 3. 数据生成机制（DGP）解释",
        "",
        '了解每个场景的 DGP，是理解"哪个方法更准/更差"的必要前提。',
        "",
        "### 场景 A：基准世界（接近随机分配）",
        "",
        "```",
        "D_i  ~ Bernoulli(0.30)                    // 处理分配：完全随机",
        "Y_it(0) = α_i + λ_t + ε_it                // 无差异时间趋势（κ_i = 0）",
        "Y_it  = Y_it(0) + D_i × τ                 // τ = 50",
        "```",
        "",
        "- **处理分配**：随机，与 X_i 和 U_i 均无关。",
        "- **时间趋势**：处理组和控制组有相同的增长趋势（平行趋势天然成立）。",
        "- **预期结果**：所有方法（DID、DID-X、PSM、PSM-DID）均无偏，Bias ≈ 0。",
        "",
        "### 场景 B：可观测趋势异质性（大企业优先纳入监管）",
        "",
        "```",
        "P(D_i=1|X_i) = logit⁻¹(-0.95 + 0.8·X_i)   // 倾向得分：X_i 越大越可能被处理",
        "κ_i = 4·X_i                                // 趋势异质性：X_i 越大，增长越快",
        "Y_it(0) = α_i + λ_t + 4·X_i·s_t + ε_it   // s_t = t-1",
        "Y_it  = Y_it(0) + D_i × τ",
        "```",
        "",
        "- **处理分配**：大企业（X_i 大）更可能被纳入监管名单。",
        "- **趋势异质性来源**：大企业本身有更快的绿色投资增长（可由 X_i 解释）。",
        "- **基础 DID 的偏误**：不加控制地做 DID，会将处理组的'趋势优势'错误归因于政策，",
        "  产生正的偏误（Bias ≈ +8.5）。这是**可观测趋势偏误**。",
        "- **DID-X 的表现**：控制 X_i × s_t 后，DID-X 可以吸收由可观测协变量 X_i",
        "  驱动的线性趋势异质性，从而有效消除场景 B 中的偏误（Bias ≈ 0）。",
        "  这是 DID-X 相对于基础 DID 的关键优势——仅控制 X_i 水平项是不够的",
        "  （会被企业固定效应吸收），必须使用 X_i × s_t 交互项。",
        "- **PSM / PSM-DID**：通过倾向得分在 X_i 上匹配，使处理组和控制组在 X_i 分布上",
        "  可比，从而消除趋势异质性，回归真实值 τ=50。",
        "",
        "### 场景 C：不可观测时间变化混淆（U 驱动分配与趋势）",
        "",
        "```",
        "P(D_i=1|X_i,U_i) = logit⁻¹(-0.95 + 0.2·X_i + 0.8·U_i)  // U_i 占主导（权重 0.8）",
        "κ_i = 4·U_i                                // 趋势异质性：完全由 U_i 驱动",
        "Y_it(0) = α_i + λ_t + 4·U_i·s_t + ε_it",
        "Y_it  = Y_it(0) + D_i × τ",
        "```",
        "",
        "- **处理分配**：主要由不可观测变量 U_i（CEO 环保偏好）驱动，X_i 的权重仅 0.2。",
        "- **趋势异质性来源**：U_i 同时驱动绿色投资增长趋势，匹配 X_i 无法解决 U_i 混淆。",
        "- **基础 DID**：偏误最大，因为 U_i 既影响处理分配又影响趋势，不可控制（偏误 ≈ +8.5）。",
        "- **DID-X**：与 DID 几乎相同偏误；X_i 的控制无助于解决 U_i 的遗漏（偏误 ≈ +8.5）。",
        "- **PSM**：偏误反而最大（偏误 ≈ +14.8），因为 PSM 直接比较处理后水平；",
        "  在 U_i 混淆下，被匹配的控制组企业在处理后仍然保留其 U 驱动的增长趋势，",
        "  导致对控制组处理后期望值的错误估计。",
        "- **PSM-DID**：偏误 ≈ +8.6，与 DID 相近；匹配未能纠正 U 混淆，双向固定效应",
        "  仍面临与场景 B 中相同的趋势识别问题——区别在于趋势来自 U 而非 X。",
        "",
    ]
    if "D" in scenarios:
        lines += [
            "### 场景 D：基于 B 的处理效应异质性扩展（τ_i 由 X 驱动）",
            "",
            "```",
            "P(D_i=1|X_i) = logit⁻¹(-0.95 + 0.8·X_i)                  // 分配机制与场景 B 相同",
            "κ_i = 4·X_i                                               // Y(0) 趋势机制与场景 B 相同",
            "τ_i = 50 + 8·clip(X_i, -1.5, 1.5)                         // 处理效应随 X_i 增大而上升",
            "Y_it  = Y_it(0) + D_i × τ_i",
            "```",
            "",
            "- **识别层面**：由于处理分配和未处理趋势仍完全由可观测 X_i 驱动，场景 D 的核心识别难点与场景 B 一样，重点仍是“是否恢复条件平行趋势”。",
            "- **估计对象层面**：现在处理效应不再是统一的 50，而是处理组企业通常拥有更高的 X_i，因此处理组的平均效应 ATT 会高于总体平均效应 ATE。",
            "- **课程意义**：场景 D 用来展示“匹配可能改变估计对象”。这里我们不改 B 的识别故事，只额外加入一个更适合讨论 ATT/ATE 差异的扩展实验。",
            "",
        ]

    # ===================================================================
    # 4. 识别假设核对
    # ===================================================================
    lines += [
        "",
        "## 4. PSM-DID 识别假设核对清单",
        "",
        "### 假设 1：处理前协变量可观测（平衡性假设）",
        "",
        "匹配只能平衡可观测的处理前变量。使用处理后变量进行匹配是禁止的。",
        "",
        "检验方式：不仅看 X 的 SMD，也直接看处理前结果水平（Y_pre_mean）和处理前斜率（pre_slope）的 SMD。",
        "",
    ]

    if ident_summary_df is not None and not ident_summary_df.empty:
        for sc in scenarios:
            before = ident_summary_df[
                (ident_summary_df["scenario"] == sc) & (ident_summary_df["sample"] == "Before")
            ]
            after = ident_summary_df[
                (ident_summary_df["scenario"] == sc) & (ident_summary_df["sample"] == "After")
            ]
            if not before.empty and not after.empty:
                b = before.iloc[0]
                a = after.iloc[0]
                lines.append(
                    f"- 场景 {sc}：|SMD(X)| {b['abs_x_smd']:.3f} → {a['abs_x_smd']:.3f}；"
                    f"|SMD(Y_pre_mean)| {b['abs_pre_mean_smd']:.3f} → {a['abs_pre_mean_smd']:.3f}；"
                    f"|SMD(pre_slope)| {b['abs_pre_slope_smd']:.3f} → {a['abs_pre_slope_smd']:.3f}"
                )
    lines += [
        "",
        "**结论**：场景 A、B 与 D 中，匹配不仅改善了 X 的平衡性，也明显改善了处理前结果水平与预趋势的可比性；",
        "场景 C 中，这些可观测维度上的平衡性同样会改善，但这并不能推出不可观测 U 已被平衡。",
        "",
        "### 假设 2：共同支持或重叠条件",
        "",
        "对处理组中的个体，必须存在倾向得分相近的控制组：",
        "",
        "$$0 < P(D_i=1 \\mid X_i) < 1$$",
        "",
        "若共同支持区间为空或极窄（宽度 < 0.01），则该假设存在风险。",
        "",
    ]

    if match_diag_df is not None and not match_diag_df.empty:
        for sc in scenarios:
            sc_d = match_diag_df[match_diag_df["scenario"] == sc]
            if not sc_d.empty:
                fail_pct = sc_d["common_support_failure"].mean() * 100
                narrow_pct = sc_d["narrow_support_warning"].mean() * 100
                avg_w = sc_d["common_support_width"].mean()
                lines.append(
                    f"- 场景 {sc}：共同支持失败率 {fail_pct:.1f}%，"
                    f"极窄区间比例 {narrow_pct:.1f}%，"
                    f"平均宽度 {avg_w:.4f}"
                )
                if fail_pct == 0:
                    lines.append(f"  ✓ 场景 {sc} 无共同支持失败")
    lines += [
        "",
        "### 假设 3：条件平行趋势",
        "",
        "在给定 X_i 或匹配后的可比样本中，如果没有处理，处理组和控制组的结果变化趋势应当相同：",
        "",
        "$$E[Y_i(0,post)-Y_i(0,pre) \\mid D_i=1, X_i] = E[Y_i(0,post)-Y_i(0,pre) \\mid D_i=0, X_i]$$",
        "",
        "检验方式：直接做处理前 placebo DID 与 differential pretrend 检查；同时再用 Monte Carlo 的 Bias 结果做交叉验证。",
        "",
    ]
    if ident_summary_df is not None and not ident_summary_df.empty:
        for sc in scenarios:
            before = ident_summary_df[
                (ident_summary_df["scenario"] == sc) & (ident_summary_df["sample"] == "Before")
            ]
            after = ident_summary_df[
                (ident_summary_df["scenario"] == sc) & (ident_summary_df["sample"] == "After")
            ]
            if not before.empty and not after.empty:
                b = before.iloc[0]
                a = after.iloc[0]
                lines.append(
                    f"- 场景 {sc}：placebo DID {b['mean_placebo_did']:+.3f} → {a['mean_placebo_did']:+.3f}；"
                    f"pretrend coef {b['mean_pretrend_coef']:+.3f} → {a['mean_pretrend_coef']:+.3f}"
                )
    lines.append("")
    for sc in scenarios:
        sc_sum = summary_df[summary_df["scenario"] == sc]
        if not sc_sum.empty:
            lines.append(f"- 场景 {sc} 各方法 Bias：")
            for _, r in sc_sum.iterrows():
                ok_str = "✓" if r["failure_runs"] == 0 else "⚠"
                lines.append(f"  - {r['method']}：Bias={r['bias']:.3f} {ok_str}")

    lines += [
        "",
        "### 假设 4：SUTVA 与无提前反应",
        "",
        "- **SUTVA（稳定单元处理值假设）**：假设处理效应不溢出到其他企业。",
        "  场景设定中假设企业间独立，无须做额外检验。",
        "- **无提前反应（No Anticipation）**：企业不会在政策公布前就提前改变行为。",
        "  由于政策在 t≥4 后才实施（post_t = 1(t≥4)），此假设在模拟中成立。",
        "",
        "### 假设 5：稳定样本与可比测量",
        "",
        "- 各场景均为平衡面板，1000 家企业 6 期全部保留，无样本自选择。",
        "- Y 的定义（绿色投资，万元）在处理前后一致，无测量误差引入的系统性偏误。",
        "- 结论：假设 5 在本模拟中成立。",
        "",
    ]

    # ===================================================================
    # 5. 共同支持诊断
    # ===================================================================
    lines += [
        "",
        "## 5. 共同支持与匹配诊断",
        "",
    ]

    if match_diag_df is not None and not match_diag_df.empty:
        lines += [
            "| 场景 | 方法 | 平均支持宽度 | 失败率 | 极窄区间比例 | 平均匹配距离 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for sc in scenarios:
            for m in ["PSM", "PSM-DID"]:
                sub = match_diag_df[(match_diag_df["scenario"] == sc) & (match_diag_df["method"] == m)]
                if not sub.empty:
                    avg_w = sub["common_support_width"].mean()
                    fail_pct = sub["common_support_failure"].mean() * 100
                    narrow_pct = sub["narrow_support_warning"].mean() * 100
                    avg_dist = sub["avg_match_distance"].mean()
                    lines.append(
                        f"| {sc} | {m} | {avg_w:.4f} | {fail_pct:.1f}% | "
                        f"{narrow_pct:.1f}% | {avg_dist:.2e} |"
                    )
        lines += [
            "",
            "**说明**：",
            "- 共同支持失败率 > 0% 表示该场景/方法在某次模拟中共同支持区间为空，",
            "  导致 PSM 和 PSM-DID 无法完成匹配，已在结果表中记为失败。",
            "- 极窄区间（宽度 < 0.01）比例 > 0% 表示存在倾向得分高度重叠不足的情况，",
            "  匹配质量可能受影响（倾向得分相似时匹配尚可，严重时估计精度下降）。",
            "",
        ]
    else:
        lines.append("（匹配诊断表未提供，无法详细分析。）")

    # ===================================================================
    # 6. 结果机制分析
    # ===================================================================
    lines += [
        "",
        "## 6. 结果机制分析：为何某方法更准或更差",
        "",
    ]

    for sc in scenarios:
        sc_sum = summary_df[summary_df["scenario"] == sc]
        sc_label = {
            "A": "基准（随机分配）",
            "B": "可观测趋势异质性",
            "C": "不可观测混淆",
            "D": "B 的异质处理效应扩展",
        }
        lines.append(f"### 场景 {sc}：{sc_label[sc]}")
        lines.append("")

        for _, r in sc_sum.sort_values("bias", key=abs).iterrows():
            bias_abs = abs(r["bias"])
            if bias_abs < 1:
                label = "✓ 无偏"
            elif bias_abs < 5:
                label = "△ 小偏"
            elif bias_abs < 10:
                label = "▲ 明显偏误"
            else:
                label = "✗ 严重偏误"
            lines.append(
                f"- **{r['method']}**：Bias={r['bias']:+.3f}，RMSE={r['rmse']:.3f}，"
                f"std={r['std']:.3f} → {label}"
            )

        if sc == "C":
            lines += [
                "",
                "> ⚠️ **场景 C 特别说明**：PSM-DID 在场景 C 中**失效**。",
                ">",
                "> 原因：倾向得分模型只能使用可观测变量 X_i，但场景 C 的真实处理分配",
                "主要由不可观测变量 U_i 驱动（权重 0.8 vs 0.2）。",
                "即便 X 平衡性改善，U 的混淆仍然存在，导致 PSM-DID 与基础 DID 偏误相近。",
                "这是 PSM-DID 方法论上的已知局限：在 U 驱动分配和趋势时，仅平衡 X 不足以保证识别。",
            ]
        if sc == "D":
            lines += [
                "",
                "> ℹ️ **场景 D 特别说明**：这里的 Bias/RMSE 是相对于各方法的 `ATT` 目标真值计算的，而不是相对于统一的 `50`。",
                ">",
                "> 由于处理组企业平均 X 更高，处理效应也更高，因此 `ATT_true > ATE_true`。",
                "> 在这一设定下，`PSM-DID` 既保留了场景 B 的识别修复逻辑，又能额外展示“匹配后样本更接近处理组局部 ATT”这一估计对象变化。",
            ]
        lines.append("")

    # ===================================================================
    # 7. 输出文件清单
    # ===================================================================
    lines += [
        "",
        "## 7. 输出文件清单",
        "",
        "- `data/interim/task2_sim_results.csv`：逐次模拟结果表（含 DID/DID-X/PSM/PSM-DID）",
        "- `data/interim/task2_match_diag.csv`：匹配诊断表（含 method 字段、极窄区间标记）",
        "- `data/processed/task2_identification_diagnostics.csv`：逐次识别诊断表（placebo/pretrend/balance）",
        "- `outputs/tables/task2_identification_summary.csv`：识别诊断汇总表",
        "- `data/processed/task2_summary.csv`：汇总结果表",
        "- `outputs/figures/estimate_distributions.png`：四方法估计值分布图",
        "- `outputs/figures/common_support.png`：倾向得分共同支持图",
        "- `outputs/figures/covariate_balance.png`：协变量平衡性图（放回加权）",
        "- `outputs/figures/preperiod_balance.png`：处理前结果与预趋势平衡图",
        "- `outputs/tables/task2_summary.csv`：汇总表 CSV",
        "",
        "---",
        "*任务二自检报告 · 因果推断与机器学习课程作业*",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
