"""
敏感性分析模块：Rosenbaum 边界检验 + 卡尺匹配对照

Rosenbaum (2002) bounds test for matched pairs.
卡尺匹配作为匹配方法的稳健性对照。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Rosenbaum 边界检验
# ---------------------------------------------------------------------------

def rosenbaum_bounds(
    treated_y: np.ndarray,
    control_y: np.ndarray,
    gamma_range: np.ndarray | None = None,
    alpha: float = 0.05,
) -> dict[str, float | np.ndarray | None]:
    """
    对匹配对计算 Rosenbaum 边界检验。

    Parameters
    ----------
    treated_y : 处理组匹配对的结果
    control_y : 配对控制组的结果
    gamma_range : 要检验的 Gamma 值数组，默认 1.0 到 3.0 步长 0.1
    alpha : 显著性水平，默认 0.05

    Returns
    -------
    dict with keys:
        gamma_star : 刚好使 p_upper >= alpha 的最小 Gamma（None 若全通过）
        gamma_range : 检验的 Gamma 数组
        p_upper : 每个 Gamma 对应的上界 p 值
        p_lower : 每个 Gamma 对应的下界 p 值
        n_pairs : 有效匹配对数（排除 exact ties）
    """
    if gamma_range is None:
        gamma_range = np.arange(1.0, 3.1, 0.1)

    d = np.asarray(treated_y, dtype=float) - np.asarray(control_y, dtype=float)

    # 排除 exact ties
    nonzero = d != 0
    d = d[nonzero]
    n = len(d)
    if n == 0:
        return {
            "gamma_star": None,
            "gamma_range": gamma_range,
            "p_upper": np.full(len(gamma_range), np.nan),
            "p_lower": np.full(len(gamma_range), np.nan),
            "n_pairs": 0,
        }

    # Wilcoxon signed-rank 统计量
    abs_d = np.abs(d)
    ranks = np.argsort(np.argsort(abs_d)) + 1  # 秩，从 1 开始
    # 处理 ties: average rank
    sorted_idx = np.argsort(abs_d)
    rank_values = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and abs_d[sorted_idx[j]] == abs_d[sorted_idx[i]]:
            j += 1
        avg_rank = (sorted_idx[i:j].mean() + 1) if j > i else (sorted_idx[i] + 1)
        # Actually compute average rank properly
        rank_sum = 0
        for k in range(i, j):
            rank_sum += (k + 1)
        avg_r = rank_sum / (j - i)
        for k in range(i, j):
            rank_values[sorted_idx[k]] = avg_r
        i = j

    # T = 正差值的秩和
    T_obs = rank_values[d > 0].sum()

    # 对每个 Gamma，计算检验统计量的上界
    p_plus = gamma_range / (1 + gamma_range)
    p_minus = 1 / (1 + gamma_range)

    # E(T) = n(n+1)/2 * p
    # Var(T) = n(n+1)(2n+1)/24 * p * (1-p)   (approximate, continuous)
    # Z = (T - E[T]) / sqrt(Var(T))
    n_plus_one = n + 1
    two_n_plus_one = 2 * n + 1

    E_base = n * n_plus_one / 4.0
    Var_base = n * n_plus_one * two_n_plus_one / 24.0

    p_upper_vals = np.empty(len(gamma_range))
    p_lower_vals = np.empty(len(gamma_range))

    for i, g in enumerate(gamma_range):
        p_u = p_plus[i]
        p_l = p_minus[i]

        # 上界：处理组有更高概率取正值 (worst case)
        E_upper = 2 * p_u * E_base
        Var_upper = 4 * p_u * (1 - p_u) * Var_base
        if Var_upper <= 0:
            z_upper = np.inf
        else:
            z_upper = (T_obs - E_upper) / np.sqrt(Var_upper)
        p_upper_vals[i] = 1 - norm.cdf(z_upper)

        # 下界：处理组有更低概率取正值
        E_lower = 2 * p_l * E_base
        Var_lower = 4 * p_l * (1 - p_l) * Var_base
        if Var_lower <= 0:
            z_lower = -np.inf
        else:
            z_lower = (T_obs - E_lower) / np.sqrt(Var_lower)
        p_lower_vals[i] = 1 - norm.cdf(z_lower)

    # 找 Gamma*：第一个使 p_upper >= alpha 的 Gamma
    gamma_star = None
    exceed_idx = np.where(p_upper_vals >= alpha)[0]
    if len(exceed_idx) > 0:
        gamma_star = float(gamma_range[exceed_idx[0]])

    return {
        "gamma_star": gamma_star,
        "gamma_range": gamma_range,
        "p_upper": p_upper_vals,
        "p_lower": p_lower_vals,
        "n_pairs": n,
    }


# ---------------------------------------------------------------------------
# 卡尺匹配
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaliperMatchResult:
    """卡尺匹配结果"""
    treated_ids: np.ndarray
    control_ids: np.ndarray
    match_distances: np.ndarray
    n_matched: int
    n_unmatched: int
    caliper: float
    common_support_failure: bool


def caliper_match(
    firm_level: pd.DataFrame,
    propensity: np.ndarray,
    caliper: float | None = None,
    caliper_sd_mult: float = 0.2,
) -> CaliperMatchResult:
    """
    1:1 卡尺最近邻匹配（放回）。

    匹配规则：只接受 |ps_treated - ps_control| < caliper 的对。
    若处理组企业在卡尺内没有匹配对象，则该企业被丢弃。

    Parameters
    ----------
    firm_level : 已完成共同支持裁剪的企业层数据
    propensity : 倾向得分数组
    caliper : 卡尺宽度，默认 None → 0.2 × sd(propensity)
    caliper_sd_mult : 卡尺 = caliper_sd_mult × sd(propensity)，仅 caliper=None 时生效

    Returns
    -------
    CaliperMatchResult
    """
    if caliper is None:
        caliper = caliper_sd_mult * float(np.std(propensity, ddof=1))

    treated_mask = firm_level["D"].values == 1
    control_mask = firm_level["D"].values == 0

    treated_idx = np.where(treated_mask)[0]
    control_idx = np.where(control_mask)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        return CaliperMatchResult(
            treated_ids=np.array([], dtype=int),
            control_ids=np.array([], dtype=int),
            match_distances=np.array([], dtype=float),
            n_matched=0,
            n_unmatched=len(treated_idx),
            caliper=caliper,
            common_support_failure=True,
        )

    p_treated = propensity[treated_idx]
    p_control = propensity[control_idx]

    treated_firm_ids = firm_level.iloc[treated_idx]["firm_id"].values
    control_firm_ids = firm_level.iloc[control_idx]["firm_id"].values

    # 排序控制组 PS 用于二分查找
    sorted_indices = np.argsort(p_control)
    p_control_sorted = p_control[sorted_indices]

    matched_treated = []
    matched_control = []
    matched_dist = []
    unmatched = 0

    for i, p_t in enumerate(p_treated):
        idx = np.searchsorted(p_control_sorted, p_t)
        n = len(p_control_sorted)

        # 找最近的
        if idx == 0:
            best_idx = 0
        elif idx == n:
            best_idx = n - 1
        else:
            left = p_control_sorted[idx - 1]
            right = p_control_sorted[idx]
            best_idx = idx if (p_t - left) >= (right - p_t) else idx - 1

        dist = abs(p_control_sorted[best_idx] - p_t)

        if dist < caliper:
            matched_treated.append(treated_firm_ids[i])
            matched_control.append(control_firm_ids[sorted_indices[best_idx]])
            matched_dist.append(dist)
        else:
            unmatched += 1

    if len(matched_treated) == 0:
        return CaliperMatchResult(
            treated_ids=np.array([], dtype=int),
            control_ids=np.array([], dtype=int),
            match_distances=np.array([], dtype=float),
            n_matched=0,
            n_unmatched=unmatched,
            caliper=caliper,
            common_support_failure=True,
        )

    return CaliperMatchResult(
        treated_ids=np.array(matched_treated),
        control_ids=np.array(matched_control),
        match_distances=np.array(matched_dist),
        n_matched=len(matched_treated),
        n_unmatched=unmatched,
        caliper=caliper,
        common_support_failure=False,
    )
