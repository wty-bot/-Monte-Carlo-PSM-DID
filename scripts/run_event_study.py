"""
事件研究图生成脚本。

基于任务一已生成的全量 Monte Carlo 数据，对每个 scenario × sim_id：
  1. 在全样本面板上估计事件研究系数（Full sample DID）
  2. 在匹配后面板上估计事件研究系数（Matched PSM-DID）

汇总 500 次模拟的系数均值与 95% 置信带，分场景绘制事件研究图。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.estimation.methods import (
    estimate_event_study,
    prepare_matching,
    _build_matched_panel,
)
from src.analysis.summarize import plot_event_study


RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["A", "B", "C", "D"]
COEFFS_PATH = INTERIM_DIR / "event_study_coeffs.csv"


def collect_event_study_coefficients() -> pd.DataFrame:
    """遍历全部场景 × 模拟，收集事件研究系数。"""
    records: list[dict[str, object]] = []

    total_sims = 0
    for sc in SCENARIOS:
        panel = pd.read_parquet(RAW_DIR / f"scenario_{sc.lower()}/task1_scenario_{sc.lower()}.parquet")
        sim_ids = sorted(panel["sim_id"].unique())
        total_sims += len(sim_ids)

    pbar = tqdm(total=total_sims, desc="Event study", unit="sim")

    for sc in SCENARIOS:
        panel = pd.read_parquet(RAW_DIR / f"scenario_{sc.lower()}/task1_scenario_{sc.lower()}.parquet")
        sim_ids = sorted(panel["sim_id"].unique())

        for sid in sim_ids:
            sim_panel = panel[panel["sim_id"] == sid].copy()

            # ---- 全样本事件研究 ----
            full_coefs = estimate_event_study(sim_panel, ref_year=3)
            for year, coef in full_coefs.items():
                records.append({
                    "scenario": sc,
                    "sim_id": sid,
                    "sample": "full",
                    "year": year,
                    "coef": coef,
                })

            # ---- 匹配样本事件研究 ----
            prep = prepare_matching(sim_panel, sc, int(sid))
            if prep.match is not None and not prep.common_support_diag.common_support_failure:
                matched_panel = _build_matched_panel(sim_panel, prep.match)
                if not matched_panel.empty and (matched_panel["D"] == 0).any():
                    matched_coefs = estimate_event_study(matched_panel, ref_year=3)
                    for year, coef in matched_coefs.items():
                        records.append({
                            "scenario": sc,
                            "sim_id": sid,
                            "sample": "matched",
                            "year": year,
                            "coef": coef,
                        })
                else:
                    # 匹配后面板无控制组
                    for year in [1, 2, 3, 4, 5, 6]:
                        records.append({
                            "scenario": sc, "sim_id": sid, "sample": "matched",
                            "year": year, "coef": np.nan,
                        })
            else:
                # 匹配失败
                for year in [1, 2, 3, 4, 5, 6]:
                    records.append({
                        "scenario": sc, "sim_id": sid, "sample": "matched",
                        "year": year, "coef": np.nan,
                    })

            pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def main() -> None:
    print("=" * 60)
    print("事件研究图生成")
    print("=" * 60)

    print("\n[1/3] 收集事件研究系数（全样本 + 匹配样本）...")
    coeffs_df = collect_event_study_coefficients()

    print(f"\n  共 {len(coeffs_df)} 条记录")
    print(f"  全样本有效 sim 数: {coeffs_df[coeffs_df['sample']=='full']['sim_id'].nunique()}")
    matched_valid = coeffs_df[(coeffs_df["sample"] == "matched") & coeffs_df["coef"].notna()]
    print(f"  匹配样本有效 sim 数: {matched_valid['sim_id'].nunique()}")

    # 保存系数
    coeffs_df.to_csv(COEFFS_PATH, index=False)
    print(f"\n  系数已保存至: {COEFFS_PATH}")

    # 打印各场景各年的系数摘要
    print("\n[2/3] 系数摘要（均值 ± 2.5%–97.5% 分位数）:")
    for sc in SCENARIOS:
        for sample in ["full", "matched"]:
            sub = coeffs_df[(coeffs_df["scenario"] == sc) & (coeffs_df["sample"] == sample)]
            valid = sub[sub["coef"].notna()]
            if valid.empty:
                continue
            summary = valid.groupby("year")["coef"].agg(
                mean="mean", lo=lambda x: np.percentile(x, 2.5), hi=lambda x: np.percentile(x, 97.5)
            )
            print(f"\n  Scenario {sc} — {sample}:")
            for yr, row in summary.iterrows():
                print(f"    year={yr}: {row['mean']:+.3f}  [{row['lo']:+.3f}, {row['hi']:+.3f}]")

    print("\n[3/3] 绘制事件研究图...")
    out_path = plot_event_study(coeffs_df)
    print(f"  图表已保存至: {out_path}")
    print("\n完成。")


if __name__ == "__main__":
    main()
