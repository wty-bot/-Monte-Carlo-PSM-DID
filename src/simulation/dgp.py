"""
DGP 模块：定义 4 个场景的数据生成过程

数据生成遵循《任务一规划文档》的统一设定。

场景 A：DID 的基准世界（处理分配随机）
场景 B：可观测趋势异质性（趋势差异由 X_i 驱动）
场景 C：不可观测的时间变化混淆（趋势差异由不可观测 U_i 驱动）
场景 D：基于 B 的处理效应异质性扩展（τ_i 由 X_i 驱动）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit  # logit^{-1}

from ..utils.seeds import make_rng


# ---------------------------------------------------------------------------
# 统一参数
# ---------------------------------------------------------------------------

N_FIRMS = 1000       # 企业数量
YEARS = [1, 2, 3, 4, 5, 6]   # 面板时期
POST_YEAR = 4        # 政策开始时点（t >= 4 为处理后）
N_SIMS = 500         # 每个场景的 Monte Carlo 次数

TAU = 50.0           # 真实处理效应（万元）
LAMBDA_SLOPE = 5.0   # 共同时间趋势 λ_t = 5 * s_t

SIGMA_ETA = 15.0     # 企业层随机扰动 η_i ~ N(0, 15²)
SIGMA_EPS = 10.0     # 个体-年扰动 ε_it ~ N(0, 10²)

SCENARIOS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# 内部数据结构（用于调试版本）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SimResultDebug:
    """
    单次模拟的完整结果（调试版本）。
    仅在开发自检阶段使用，不写入正式交付数据。
    """
    scenario: str
    sim_id: int
    df: pd.DataFrame  # 含 Y(0) / Y(1) / U_i 等调试字段


@dataclass(frozen=True)
class ScenarioTruth:
    """
    单次模拟下各场景可用于结果解释的真值口径。
    """
    scenario: str
    sim_id: int
    ate_true: float
    att_true: float
    atc_true: float
    tau_sd_true: float
    tau_min_true: float
    tau_max_true: float
    n_treated: int
    n_control: int


# ---------------------------------------------------------------------------
# 核心生成函数
# ---------------------------------------------------------------------------

def _logit(x: np.ndarray) -> np.ndarray:
    """logit 逆函数：expit，即 1 / (1 + exp(-x))"""
    return expit(x)


def heterogeneous_tau_from_x(x: np.ndarray) -> np.ndarray:
    """
    场景 D 的异质处理效应：
        τ_i = 50 + 8 * clip(X_i, -1.5, 1.5)
    """
    return TAU + 8.0 * np.clip(x, -1.5, 1.5)


def compute_true_effect_stats(
    scenario: str,
    sim_id: int,
    x: np.ndarray,
    d: np.ndarray,
) -> ScenarioTruth:
    """
    计算单次模拟的真值口径。

    A/B/C 为常数处理效应，因此 ATE = ATT = ATC = 50。
    D 为异质处理效应，因此需要分别汇报总体 ATE 与处理组 ATT。
    """
    if scenario == "D":
        tau_i = heterogeneous_tau_from_x(x)
    else:
        tau_i = np.full_like(x, TAU, dtype=float)

    treated_mask = d == 1
    control_mask = d == 0
    att_true = float(tau_i[treated_mask].mean()) if treated_mask.any() else np.nan
    atc_true = float(tau_i[control_mask].mean()) if control_mask.any() else np.nan

    return ScenarioTruth(
        scenario=scenario,
        sim_id=sim_id,
        ate_true=float(tau_i.mean()),
        att_true=att_true,
        atc_true=atc_true,
        tau_sd_true=float(np.std(tau_i, ddof=0)),
        tau_min_true=float(tau_i.min()),
        tau_max_true=float(tau_i.max()),
        n_treated=int(treated_mask.sum()),
        n_control=int(control_mask.sum()),
    )


def generate_one_sim(
    scenario: str,
    sim_id: int,
) -> _SimResultDebug:
    """
    对指定场景生成一次完整的面板数据。

    Parameters
    ----------
    scenario : str
        "A"、"B"、"C" 或 "D"
    sim_id : int
        模拟编号，从 1 开始

    Returns
    -------
    _SimResultDebug
        含调试字段的完整模拟结果
    """
    rng = make_rng(scenario, sim_id)

    # ----- 个体层面变量 -----
    X = rng.standard_normal(N_FIRMS)          # X_i ~ N(0,1)
    eta = rng.normal(0, SIGMA_ETA, N_FIRMS)  # η_i ~ N(0, 15²)

    # 场景 A/B/D：α_i = 100 + 10*X_i + η_i
    # 场景 C：α_i = 100 + 10*X_i + 5*U_i + η_i
    alpha = 100.0 + 10.0 * X + eta

    U: np.ndarray | None = None
    if scenario == "C":
        U = rng.standard_normal(N_FIRMS)     # U_i ~ N(0,1)，仅场景 C
        alpha = alpha + 5.0 * U              # 更新 α_i

    # ----- 处理分配 D_i -----
    if scenario == "A":
        D = (rng.random(N_FIRMS) < 0.30).astype(np.int8)
    elif scenario in {"B", "D"}:
        p = _logit(-0.95 + 0.8 * X)
        D = (rng.random(N_FIRMS) < p).astype(np.int8)
    elif scenario == "C":
        p = _logit(-0.95 + 0.2 * X + 0.8 * U)
        D = (rng.random(N_FIRMS) < p).astype(np.int8)
    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")

    # ----- 潜在结果 Y(0) / Y(1) -----
    # 共同时间趋势 λ_t = 5 * (t - 1)
    time_trend = np.array([LAMBDA_SLOPE * (t - 1) for t in YEARS])

    # 趋势驱动项 κ_i（统一转为数组 shape (N_FIRMS,)）
    if scenario == "A":
        kappa = np.zeros(N_FIRMS)
    elif scenario in {"B", "D"}:
        kappa = 4.0 * X
    elif scenario == "C":
        kappa = 4.0 * U  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")

    # s_t = t - 1
    s = np.array([t - 1 for t in YEARS])

    # Y_it(0) = α_i + λ_t + κ_i * s_t + ε_it
    eps = rng.normal(0, SIGMA_EPS, (N_FIRMS, len(YEARS)))  # ε_it
    Y0 = alpha[:, None] + time_trend[None, :] + (kappa[:, None] * s[None, :]) + eps

    # Y_it(1) = Y_it(0) + τ_i
    if scenario == "D":
        tau_i = heterogeneous_tau_from_x(X)
    else:
        tau_i = np.full(N_FIRMS, TAU)
    Y1 = Y0 + tau_i[:, None]

    # ----- 观测结果 Y_it -----
    # Y_it = Y_it(0) + D_i * τ_i * post_t
    post = np.array([1 if t >= POST_YEAR else 0 for t in YEARS])
    Y_obs = Y0 + (D[:, None] * tau_i[:, None] * post[None, :])

    truth = compute_true_effect_stats(scenario, sim_id, X, D)

    # ----- 打包为 DataFrame -----
    records = []
    for i in range(N_FIRMS):
        for j, t in enumerate(YEARS):
            row: dict[str, int | float | str] = {
                "scenario": scenario,
                "sim_id": sim_id,
                "firm_id": i,
                "year": t,
                "post": post[j],
                "D": D[i],
                "X": X[i],
                "Y": Y_obs[i, j],
                # 调试字段（正式版本不输出）
                "Y0": Y0[i, j],
                "Y1": Y1[i, j],
                "tau_i": tau_i[i],
                "ate_true": truth.ate_true,
                "att_true": truth.att_true,
                "atc_true": truth.atc_true,
            }
            if U is not None:
                row["U"] = U[i]
            records.append(row)

    df = pd.DataFrame(records)
    # 字段顺序：主字段在前，调试字段在后
    return _SimResultDebug(scenario=scenario, sim_id=sim_id, df=df)


def run_scenario(
    scenario: str,
    n_sims: int = N_SIMS,
    progress: bool = True,
) -> pd.DataFrame:
    """
    运行指定场景的完整 Monte Carlo 循环。

    Parameters
    ----------
    scenario : str
        "A"、"B"、"C" 或 "D"
    n_sims : int
        模拟次数，默认 500
    progress : bool
        是否打印进度（每 100 次报告一次）

    Returns
    -------
    pd.DataFrame
        合并所有 sim_id 的 DataFrame（调试版本，含 Y(0)/Y(1)/U_i）
    """
    chunks: list[pd.DataFrame] = []
    for k in range(1, n_sims + 1):
        res = generate_one_sim(scenario, k)
        chunks.append(res.df)
        if progress and k % 100 == 0:
            print(f"  场景 {scenario}: 完成 {k}/{n_sims}")
    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# 正式交付版本生成（去除调试字段）
# ---------------------------------------------------------------------------

# 正式交付字段（不含调试字段）
FINAL_COLS = ["scenario", "sim_id", "firm_id", "year", "post", "D", "X", "Y"]
DEBUG_COLS = ["Y0", "Y1", "U", "tau_i", "ate_true", "att_true", "atc_true"]


def make_final_df(debug_df: pd.DataFrame) -> pd.DataFrame:
    """
    将调试版本 DataFrame 转换为正式交付版本（去除 Y0/Y1/U 等调试字段）。
    """
    debug_present = [c for c in DEBUG_COLS if c in debug_df.columns]
    final_df = debug_df.drop(columns=debug_present, errors="ignore")
    if "scenario" not in final_df.columns:
        raise ValueError("debug_df is missing required column: scenario")
    return final_df.loc[:, FINAL_COLS]
