"""
simulation包：任务一相关代码（DGP、场景定义、Monte Carlo循环）
"""

from .dgp import (
    N_FIRMS,
    YEARS,
    POST_YEAR,
    N_SIMS,
    TAU,
    LAMBDA_SLOPE,
    SIGMA_ETA,
    SIGMA_EPS,
    SCENARIOS,
    FINAL_COLS,
    DEBUG_COLS,
    ScenarioTruth,
    heterogeneous_tau_from_x,
    compute_true_effect_stats,
    generate_one_sim,
    run_scenario,
    make_final_df,
)

__all__ = [
    "N_FIRMS",
    "YEARS",
    "POST_YEAR",
    "N_SIMS",
    "TAU",
    "LAMBDA_SLOPE",
    "SIGMA_ETA",
    "SIGMA_EPS",
    "SCENARIOS",
    "FINAL_COLS",
    "DEBUG_COLS",
    "ScenarioTruth",
    "heterogeneous_tau_from_x",
    "compute_true_effect_stats",
    "generate_one_sim",
    "run_scenario",
    "make_final_df",
]
