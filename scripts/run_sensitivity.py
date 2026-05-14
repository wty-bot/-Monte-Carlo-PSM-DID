"""
敏感性分析独立脚本：Rosenbaum 边界检验 + 卡尺匹配对照

不修改任务一/任务二脚本，不重跑 DGP。
读取已有 parquet 数据，重新跑匹配部分，输出敏感性分析结果。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import raw_path, DATA_PROCESSED, OUTPUT_CHECKS, ensure_dirs
from src.estimation.methods import (
    _build_firm_level,
    _estimate_propensity,
    _common_support_trim,
    _nearest_neighbor_match,
    _twfe_coef,
    _build_matched_panel,
)
from src.analysis.sensitivity import rosenbaum_bounds, caliper_match

ensure_dirs()

SCENARIOS = ["A", "B", "C", "D"]
GAMMA_RANGE = np.arange(1.0, 3.1, 0.1)
ALPHA = 0.05


def run_one_sim_sensitivity(
    panel: pd.DataFrame,
    scenario: str,
    sim_id: int,
) -> dict:
    """
    对单次模拟运行全部敏感性分析。
    返回一行汇总字典。
    """
    firm = _build_firm_level(panel)
    p = _estimate_propensity(firm)
    trimmed_firm, trimmed_p, cs_diag = _common_support_trim(firm, p)

    row = {
        "scenario": scenario,
        "sim_id": sim_id,
        "cs_failure": cs_diag.common_support_failure,
        "cs_width": cs_diag.common_support_width,
    }

    # ---- 1:1 最近邻匹配（标准方法）----
    if cs_diag.common_support_failure:
        row.update({
            "nn_n_pairs": 0,
            "psm_gamma_star": np.nan,
            "psmdid_gamma_star": np.nan,
            "caliper_cs_failure": True,
            "caliper_n_matched": 0,
            "caliper_n_unmatched": 0,
            "psm_caliper_est": np.nan,
            "psmdid_caliper_est": np.nan,
        })
        return row

    nn_match = _nearest_neighbor_match(trimmed_firm, trimmed_p, method="sensitivity")
    row["nn_n_pairs"] = nn_match.n_treated

    # 计算匹配后样本的真实 ATT（用于 Rosenbaum 中心化）
    if scenario == "D":
        from src.simulation.dgp import heterogeneous_tau_from_x
        matched_tau = heterogeneous_tau_from_x(
            trimmed_firm.loc[trimmed_firm["firm_id"].isin(nn_match.treated_ids), "X"].values
        )
        true_att_matched = float(np.mean(matched_tau))
    else:
        true_att_matched = 50.0

    # Rosenbaum 检验中心化在真实 ATT 上：检验"偏离真值是否可由隐藏偏误解释"
    # 标准用法检验"效应是否为零"，我们检验"残余偏误是否为零"

    # Rosenbaum PSM：匹配对的 (Y_post_mean 差 - 真值)，即检验 PSM 的偏误
    treated_post = trimmed_firm.loc[
        trimmed_firm["firm_id"].isin(nn_match.treated_ids), ["firm_id", "Y_post_mean"]
    ].set_index("firm_id")["Y_post_mean"]
    ctrl_post = trimmed_firm.loc[
        trimmed_firm["firm_id"].isin(nn_match.control_ids), ["firm_id", "Y_post_mean"]
    ].set_index("firm_id")["Y_post_mean"]
    treated_vals = np.array([treated_post.loc[fid] for fid in nn_match.treated_ids])
    control_vals = np.array([ctrl_post.loc[fid] for fid in nn_match.control_ids])
    rb_psm = rosenbaum_bounds(treated_vals - true_att_matched, control_vals, GAMMA_RANGE, ALPHA)
    row["psm_gamma_star"] = rb_psm["gamma_star"] if rb_psm["gamma_star"] is not None else np.nan

    # Rosenbaum PSM-DID：匹配对的 (DeltaY 差 - 真值)
    treated_dy = trimmed_firm.loc[
        trimmed_firm["firm_id"].isin(nn_match.treated_ids), ["firm_id", "DeltaY"]
    ].set_index("firm_id")["DeltaY"]
    ctrl_dy = trimmed_firm.loc[
        trimmed_firm["firm_id"].isin(nn_match.control_ids), ["firm_id", "DeltaY"]
    ].set_index("firm_id")["DeltaY"]
    treated_dy_vals = np.array([treated_dy.loc[fid] for fid in nn_match.treated_ids])
    ctrl_dy_vals = np.array([ctrl_dy.loc[fid] for fid in nn_match.control_ids])
    rb_psmdid = rosenbaum_bounds(treated_dy_vals - true_att_matched, ctrl_dy_vals, GAMMA_RANGE, ALPHA)
    row["psmdid_gamma_star"] = (
        rb_psmdid["gamma_star"] if rb_psmdid["gamma_star"] is not None else np.nan
    )

    # ---- 卡尺匹配 ----
    caliper_result = caliper_match(trimmed_firm, trimmed_p)
    row["caliper_cs_failure"] = caliper_result.common_support_failure
    row["caliper_n_matched"] = caliper_result.n_matched
    row["caliper_n_unmatched"] = caliper_result.n_unmatched
    row["caliper_width"] = caliper_result.caliper

    if not caliper_result.common_support_failure and caliper_result.n_matched > 0:
        # PSM-Caliper：卡尺匹配对的 Y_post_mean 差
        cal_treated_post = trimmed_firm.loc[
            trimmed_firm["firm_id"].isin(caliper_result.treated_ids), ["firm_id", "Y_post_mean"]
        ].set_index("firm_id")["Y_post_mean"]
        cal_ctrl_post = trimmed_firm.loc[
            trimmed_firm["firm_id"].isin(caliper_result.control_ids), ["firm_id", "Y_post_mean"]
        ].set_index("firm_id")["Y_post_mean"]
        cal_t_vals = np.array([cal_treated_post.loc[fid] for fid in caliper_result.treated_ids])
        cal_c_vals = np.array([cal_ctrl_post.loc[fid] for fid in caliper_result.control_ids])
        row["psm_caliper_est"] = float(np.mean(cal_t_vals - cal_c_vals))

        # PSM-DID-Caliper：卡尺匹配后 TWFE
        from src.estimation.methods import MatchResult
        cal_match_for_panel = MatchResult(
            method="PSM-DID-Caliper",
            treated_ids=caliper_result.treated_ids,
            control_ids=caliper_result.control_ids,
            match_distances=caliper_result.match_distances,
            n_treated=caliper_result.n_matched,
            n_control_unique=len(np.unique(caliper_result.control_ids)),
            common_support_min=cs_diag.common_support_min,
            common_support_max=cs_diag.common_support_max,
            common_support_width=cs_diag.common_support_width,
            common_support_failure=False,
            narrow_support_warning=False,
            n_before_trim_treated=cs_diag.n_before_trim_treated,
            n_before_trim_control=cs_diag.n_before_trim_control,
            n_after_trim_treated=cs_diag.n_after_trim_treated,
            n_after_trim_control=cs_diag.n_after_trim_control,
        )
        cal_matched_panel = _build_matched_panel(panel, cal_match_for_panel)
        cal_matched_panel["D_post"] = cal_matched_panel["D"] * cal_matched_panel["post"]
        row["psmdid_caliper_est"] = _twfe_coef(cal_matched_panel, treatment_col="D_post")
    else:
        row["psm_caliper_est"] = np.nan
        row["psmdid_caliper_est"] = np.nan

    return row


def main() -> None:
    all_rows = []

    for sc in SCENARIOS:
        path = raw_path(sc, kind="final")
        if not path.exists():
            print(f"  SKIP scenario {sc}: {path} not found")
            continue

        print(f"Loading scenario {sc} ...", flush=True)
        df = pd.read_parquet(path)

        sim_ids = sorted(df["sim_id"].unique())
        n_total = len(sim_ids)
        for i, sid in enumerate(sim_ids):
            panel = df[df["sim_id"] == sid].copy()
            row = run_one_sim_sensitivity(panel, sc, sid)
            all_rows.append(row)
            if (i + 1) % 50 == 0:
                print(f"  [{sc}] {i+1}/{n_total} sims done", flush=True)

        print(f"  [{sc}] complete: {n_total} sims", flush=True)

    # ---- 保存 ----
    sim_df = pd.DataFrame(all_rows)
    sim_path = DATA_PROCESSED / "task2_sensitivity_sim.csv"
    sim_df.to_csv(sim_path, index=False)
    print(f"Saved: {sim_path}", flush=True)

    # ---- 汇总 ----
    summary_rows = []
    for sc in SCENARIOS:
        sc_df = sim_df[sim_df["scenario"] == sc]
        # Rosenbaum
        psm_g = sc_df["psm_gamma_star"].dropna()
        psmdid_g = sc_df["psmdid_gamma_star"].dropna()
        # Caliper estimates
        psm_cal = sc_df["psm_caliper_est"].dropna()
        psmdid_cal = sc_df["psmdid_caliper_est"].dropna()
        # Unmatch rate for caliper
        cal_fail_rate = sc_df["caliper_cs_failure"].mean() if len(sc_df) > 0 else np.nan
        cal_avg_unmatched = sc_df.loc[~sc_df["caliper_cs_failure"], "caliper_n_unmatched"].mean() if (~sc_df["caliper_cs_failure"]).sum() > 0 else np.nan

        summary_rows.append({
            "scenario": sc,
            "method": "PSM",
            "analysis": "Rosenbaum",
            "mean_gamma_star": psm_g.mean() if len(psm_g) > 0 else np.nan,
            "median_gamma_star": psm_g.median() if len(psm_g) > 0 else np.nan,
            "pct_gamma_star_le_1_5": (psm_g <= 1.5).mean() * 100 if len(psm_g) > 0 else np.nan,
            "valid_runs": len(psm_g),
        })
        summary_rows.append({
            "scenario": sc,
            "method": "PSM-DID",
            "analysis": "Rosenbaum",
            "mean_gamma_star": psmdid_g.mean() if len(psmdid_g) > 0 else np.nan,
            "median_gamma_star": psmdid_g.median() if len(psmdid_g) > 0 else np.nan,
            "pct_gamma_star_le_1_5": (psmdid_g <= 1.5).mean() * 100 if len(psmdid_g) > 0 else np.nan,
            "valid_runs": len(psmdid_g),
        })
        summary_rows.append({
            "scenario": sc,
            "method": "PSM-Caliper",
            "analysis": "Caliper",
            "mean_gamma_star": np.nan,
            "median_gamma_star": np.nan,
            "pct_gamma_star_le_1_5": cal_fail_rate * 100,
            "valid_runs": len(psm_cal),
            "extra_info": f"avg_unmatched={cal_avg_unmatched:.1f}" if not np.isnan(cal_avg_unmatched) else "",
        })
        summary_rows.append({
            "scenario": sc,
            "method": "PSM-DID-Caliper",
            "analysis": "Caliper",
            "mean_gamma_star": np.nan,
            "median_gamma_star": np.nan,
            "pct_gamma_star_le_1_5": np.nan,
            "valid_runs": len(psmdid_cal),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = DATA_PROCESSED / "task2_sensitivity_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # ---- 打印核心结果 ----
    print("\n=== Rosenbaum Gamma* Summary ===")
    for sc in SCENARIOS:
        sc_df = sim_df[sim_df["scenario"] == sc]
        print(f"\nScenario {sc}:")
        for method, col in [("PSM", "psm_gamma_star"), ("PSM-DID", "psmdid_gamma_star")]:
            g = sc_df[col].dropna()
            if len(g) == 0:
                print(f"  {method}: no valid runs")
            else:
                print(f"  {method}: mean={g.mean():.2f}, median={g.median():.2f}, "
                      f"pct(Gamma*<=1.5)={(g <= 1.5).mean() * 100:.1f}%")

    print("\n=== Caliper Match Summary ===")
    for sc in SCENARIOS:
        sc_df = sim_df[sim_df["scenario"] == sc]
        print(f"\nScenario {sc}:")
        for method, col in [("PSM-Caliper", "psm_caliper_est"), ("PSM-DID-Caliper", "psmdid_caliper_est")]:
            est = sc_df[col].dropna()
            if len(est) == 0:
                print(f"  {method}: no valid runs")
            else:
                bias = est.mean() - 50
                print(f"  {method}: n_valid={len(est)}, mean_est={est.mean():.3f}, bias={bias:+.3f}, std={est.std():.3f}")


if __name__ == "__main__":
    main()
