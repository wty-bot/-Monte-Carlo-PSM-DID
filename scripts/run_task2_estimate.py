"""
任务二主估计脚本：PSM、DID、PSM-DID 的估计与比较

用法：
    python scripts/run_task2_estimate.py [--n-sims N] [--smoke]

参数：
    --n-sims    每个场景的模拟次数（默认 500）
    --smoke     烟雾测试模式，每场景仅运行 10 次
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# 确保项目根在路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.estimation.methods import run_one_simulation
from src.analysis.summarize import (
    summarize_estimates,
    compute_identification_diagnostics_for_sim,
    summarize_identification_diagnostics,
    plot_estimate_distributions,
    plot_common_support,
    plot_covariate_balance,
    plot_preperiod_balance,
    generate_task2_check_report,
)
from src.simulation.dgp import SCENARIOS as ALL_SCENARIOS
from src.utils.paths import (
    ensure_dirs,
    raw_path,
    DATA_INTERIM,
    DATA_PROCESSED,
    OUTPUT_FIGURES,
    OUTPUT_TABLES,
    OUTPUT_CHECKS,
    OUTPUT_LOGS,
)

SCENARIOS = ALL_SCENARIOS
PROGRESS_PATH = OUTPUT_LOGS / "task2_progress.json"


def write_progress(payload: dict) -> None:
    """写入任务二运行进度，供外部实时查看。"""
    ensure_dirs()
    PROGRESS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_scenario_data(scenario: str) -> pd.DataFrame:
    """加载任务一生成的正式数据"""
    p = raw_path(scenario, kind="final")
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在: {p}")
    return pd.read_parquet(p)


def parse_scenarios_arg(scenarios_text: str | None) -> list[str]:
    """解析命令行传入的场景列表。"""
    if not scenarios_text:
        return list(SCENARIOS)
    scenarios = [item.strip().upper() for item in scenarios_text.split(",") if item.strip()]
    invalid = [sc for sc in scenarios if sc not in SCENARIOS]
    if invalid:
        raise ValueError(f"未知场景: {invalid}，可选值为 {SCENARIOS}")
    return scenarios


def run_all_estimations(
    n_sims: int = 500,
    smoke: bool = False,
    scenarios: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    对指定场景 × n_sims 次模拟执行四种方法的估计。

    Returns
    -------
    results_df : 逐次模拟结果表
    match_diag_df : 匹配诊断表
    ident_diag_df : 逐次识别诊断表
    """
    all_results: list[dict] = []
    all_match_diag: list[dict] = []
    all_ident_diag: list[dict] = []
    scenario_list = scenarios or list(SCENARIOS)
    grand_total = len(scenario_list) * (10 if smoke else n_sims)
    grand_completed = 0
    run_started_at = time.time()

    for idx, sc in enumerate(scenario_list):
        print(f"\n{'='*60}")
        print(f"  场景 {sc}：加载数据并执行估计")
        print(f"{'='*60}")

        t0 = time.time()
        panel = load_scenario_data(sc)
        sim_ids = sorted(panel["sim_id"].unique())

        if smoke:
            sim_ids = sim_ids[:10]
            print(f"  [烟雾测试] 仅运行前 {len(sim_ids)} 次模拟")
        else:
            sim_ids = sim_ids[:n_sims]
        scenario_total = len(sim_ids)
        write_progress({
            "status": "running",
            "mode": "smoke" if smoke else "full",
            "current_stage": "estimation",
            "current_scenario": sc,
            "scenario_completed": 0,
            "scenario_total": scenario_total,
            "overall_completed": grand_completed,
            "overall_total": grand_total,
            "elapsed_seconds": round(time.time() - run_started_at, 1),
            "message": f"场景 {sc} 开始运行",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        desc = f"场景 {sc}"
        for sim_id in tqdm(sim_ids, desc=desc, unit="sim", ncols=80):
            sim_panel = panel[panel["sim_id"] == sim_id].copy()

            estimates, matches, prep = run_one_simulation(sim_panel, sc, sim_id)

            for est in estimates:
                all_results.append({
                    "scenario": est.scenario,
                    "sim_id": est.sim_id,
                    "method": est.method,
                    "estimate": est.estimate,
                    "bias": est.bias,
                    "sq_error": est.sq_error,
                    "status": est.status,
                    "target_estimand": est.target_estimand,
                    "true_effect_target": est.true_effect_target,
                    "n_matched_pairs": est.n_matched_pairs,
                    "n_control_unique": est.n_control_unique,
                    "avg_match_distance": est.avg_match_distance,
                    "common_support_width": est.common_support_width,
                    "n_before_trim": est.n_before_trim,
                    "n_after_trim": est.n_after_trim,
                })

            for m in matches:
                all_match_diag.append({
                    "scenario": sc,
                    "sim_id": sim_id,
                    "method": m.method,  # "PSM" or "PSM-DID"
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

            all_ident_diag.extend(
                compute_identification_diagnostics_for_sim(sim_panel, sc, sim_id, prep)
            )
            grand_completed += 1
            scenario_completed = grand_completed - (idx * scenario_total)
            if (
                scenario_completed == scenario_total
                or scenario_completed == 1
                or scenario_completed % 25 == 0
            ):
                write_progress({
                    "status": "running",
                    "mode": "smoke" if smoke else "full",
                    "current_stage": "estimation",
                    "current_scenario": sc,
                    "scenario_completed": scenario_completed,
                    "scenario_total": scenario_total,
                    "overall_completed": grand_completed,
                    "overall_total": grand_total,
                    "elapsed_seconds": round(time.time() - run_started_at, 1),
                    "message": f"场景 {sc} 已完成 {scenario_completed}/{scenario_total}",
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })

        elapsed = time.time() - t0
        print(f"  场景 {sc} 全部完成，共 {len(sim_ids)} 次，耗时 {elapsed:.1f}s")

    results_df = pd.DataFrame(all_results)
    match_diag_df = pd.DataFrame(all_match_diag)
    ident_diag_df = pd.DataFrame(all_ident_diag)
    return results_df, match_diag_df, ident_diag_df


def save_results(
    results_df: pd.DataFrame,
    match_diag_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ident_diag_df: pd.DataFrame,
    ident_summary_df: pd.DataFrame,
):
    """保存所有结果到磁盘"""
    ensure_dirs()

    # 中间结果
    results_path = DATA_INTERIM / "task2_sim_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"  逐次模拟结果已保存：{results_path}")

    if not match_diag_df.empty:
        diag_path = DATA_INTERIM / "task2_match_diag.csv"
        match_diag_df.to_csv(diag_path, index=False)
        print(f"  匹配诊断表已保存：{diag_path}")

    ident_path = DATA_PROCESSED / "task2_identification_diagnostics.csv"
    ident_diag_df.to_csv(ident_path, index=False)
    print(f"  逐次识别诊断表已保存：{ident_path}")

    # 最终汇总
    summary_csv_proc = DATA_PROCESSED / "task2_summary.csv"
    summary_df.to_csv(summary_csv_proc, index=False)
    print(f"  汇总结果表已保存：{summary_csv_proc}")

    ident_summary_proc = OUTPUT_TABLES / "task2_identification_summary.csv"
    ident_summary_df.to_csv(ident_summary_proc, index=False)
    print(f"  识别诊断汇总表已保存：{ident_summary_proc}")

    # 汇总表副本到 outputs/tables
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    summary_csv_out = OUTPUT_TABLES / "task2_summary.csv"
    summary_df.to_csv(summary_csv_out, index=False)
    print(f"  汇总表副本已保存：{summary_csv_out}")


def main():
    parser = argparse.ArgumentParser(description="任务二：PSM、DID、PSM-DID 估计与比较")
    parser.add_argument("--n-sims", type=int, default=500)
    parser.add_argument("--smoke", action="store_true", help="烟雾测试模式（每场景 10 次）")
    parser.add_argument("--scenarios", type=str, default=None, help="逗号分隔场景，如 A,B,C,D 或 D")
    args = parser.parse_args()
    scenario_list = parse_scenarios_arg(args.scenarios)

    print(f"\n任务二主脚本 — Python {sys.version.split()[0]}")
    write_progress({
        "status": "starting",
        "mode": "smoke" if args.smoke else "full",
        "current_stage": "initializing",
        "current_scenario": None,
        "scenario_completed": 0,
        "scenario_total": 0,
        "overall_completed": 0,
        "overall_total": 10 * len(scenario_list) if args.smoke else args.n_sims * len(scenario_list),
        "elapsed_seconds": 0.0,
        "message": "任务二脚本启动",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    if args.smoke:
        print("[烟雾测试模式] 每场景运行 10 次模拟")
        n_sims = 10
    else:
        n_sims = args.n_sims
        print(f"每个场景模拟次数：{n_sims}")
    print(f"本次运行场景：{', '.join(scenario_list)}")

    # 1. 执行估计
    t_start = time.time()
    results_df, match_diag_df, ident_diag_df = run_all_estimations(
        n_sims=n_sims,
        smoke=args.smoke,
        scenarios=scenario_list,
    )
    elapsed = time.time() - t_start
    print(f"\n全部估计完成，总耗时 {elapsed:.1f}s")

    # 2. 汇总
    write_progress({
        "status": "running",
        "mode": "smoke" if args.smoke else "full",
        "current_stage": "summarizing",
        "current_scenario": None,
        "scenario_completed": n_sims,
        "scenario_total": n_sims,
        "overall_completed": len(scenario_list) * n_sims,
        "overall_total": len(scenario_list) * n_sims,
        "elapsed_seconds": round(elapsed, 1),
        "message": "主估计完成，开始汇总结果",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print("\n生成汇总表 ...")
    summary_df = summarize_estimates(results_df)
    print("\n汇总结果：")
    print(summary_df.to_string(index=False))

    # 3. 生成图表与识别诊断
    print("\n生成图表 ...")
    # 需要加载原始面板数据用于共同支持图和平衡性图
    panel_for_plots = {}
    for sc in scenario_list:
        sc_panel = load_scenario_data(sc)
        keep_sim_ids = sorted(sc_panel["sim_id"].unique())[:n_sims]
        panel_for_plots[sc] = sc_panel[sc_panel["sim_id"].isin(keep_sim_ids)].copy()
    full_panel = pd.concat(panel_for_plots.values(), ignore_index=True)

    print("\n汇总直接识别诊断 ...")
    ident_summary_df = summarize_identification_diagnostics(ident_diag_df)
    print(ident_summary_df.to_string(index=False))

    fig1 = plot_estimate_distributions(results_df, out_dir=OUTPUT_FIGURES)
    print(f"  估计值分布图已保存：{fig1}")

    fig2 = plot_common_support(full_panel, out_dir=OUTPUT_FIGURES)
    print(f"  共同支持图已保存：{fig2}")

    fig3 = plot_covariate_balance(ident_diag_df, out_dir=OUTPUT_FIGURES)
    print(f"  平衡性图已保存：{fig3}")

    fig4 = plot_preperiod_balance(ident_diag_df, out_dir=OUTPUT_FIGURES)
    print(f"  处理前平衡性图已保存：{fig4}")

    # 4. 自检报告
    write_progress({
        "status": "running",
        "mode": "smoke" if args.smoke else "full",
        "current_stage": "reporting",
        "current_scenario": None,
        "scenario_completed": n_sims,
        "scenario_total": n_sims,
        "overall_completed": len(scenario_list) * n_sims,
        "overall_total": len(scenario_list) * n_sims,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "message": "图表与识别诊断完成，开始生成自检报告",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print("\n生成自检报告 ...")
    report_path = generate_task2_check_report(
        summary_df,
        results_df,
        match_diag_df,
        ident_summary_df=ident_summary_df,
        out_dir=OUTPUT_CHECKS,
    )
    print(f"  自检报告已保存：{report_path}")

    # 5. 保存结果
    write_progress({
        "status": "running",
        "mode": "smoke" if args.smoke else "full",
        "current_stage": "saving",
        "current_scenario": None,
        "scenario_completed": n_sims,
        "scenario_total": n_sims,
        "overall_completed": len(scenario_list) * n_sims,
        "overall_total": len(scenario_list) * n_sims,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "message": "自检报告完成，开始保存结果文件",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print("\n保存结果文件 ...")
    save_results(results_df, match_diag_df, summary_df, ident_diag_df, ident_summary_df)

    write_progress({
        "status": "completed",
        "mode": "smoke" if args.smoke else "full",
        "current_stage": "done",
        "current_scenario": None,
        "scenario_completed": n_sims,
        "scenario_total": n_sims,
        "overall_completed": len(scenario_list) * n_sims,
        "overall_total": len(scenario_list) * n_sims,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "message": "任务二全部完成",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    print(f"\n{'='*60}")
    print("  任务二全部完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
