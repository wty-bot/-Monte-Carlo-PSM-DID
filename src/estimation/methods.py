"""
估计模块：PSM、DID、PSM-DID 的实现

所有方法遵循《任务二规划文档》的统一口径。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import expit
import statsmodels.api as sm

from ..simulation.dgp import compute_true_effect_stats, heterogeneous_tau_from_x, TAU


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchResult:
    """单次匹配的结果"""
    method: str                  # "PSM" or "PSM-DID"
    treated_ids: np.ndarray       # 处理组 firm_id
    control_ids: np.ndarray       # 对应匹配到的控制组 firm_id
    match_distances: np.ndarray   # 匹配距离
    n_treated: int
    n_control_unique: int         # 去重后的控制组企业数
    common_support_min: float
    common_support_max: float
    common_support_width: float
    common_support_failure: bool
    narrow_support_warning: bool   # 共同支持宽度 < 0.01 的警告
    n_before_trim_treated: int
    n_before_trim_control: int
    n_after_trim_treated: int
    n_after_trim_control: int
    scenario: str = ""
    sim_id: int | None = None


@dataclass(frozen=True)
class EstimationResult:
    """单次模拟的估计结果"""
    scenario: str
    sim_id: int
    method: str                   # "PSM", "DID", "PSM-DID"
    estimate: float | None
    bias: float | None
    sq_error: float | None
    status: str                   # "ok" or "common_support_failure"
    # 匹配诊断（仅 PSM / PSM-DID 有值）
    n_matched_pairs: int | None = None
    n_control_unique: int | None = None
    avg_match_distance: float | None = None
    common_support_width: float | None = None
    n_before_trim: int | None = None
    n_after_trim: int | None = None
    target_estimand: str | None = None
    true_effect_target: float | None = None


@dataclass(frozen=True)
class MatchingPrep:
    """PSM / PSM-DID 共用的匹配准备结果。"""
    scenario: str
    sim_id: int
    firm_level: pd.DataFrame
    propensity: np.ndarray
    trimmed_firm: pd.DataFrame
    trimmed_propensity: np.ndarray
    common_support_diag: MatchResult
    match: MatchResult | None
    true_effect_stats: dict[str, float | int]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _build_firm_level(panel: pd.DataFrame) -> pd.DataFrame:
    """
    从 6 期面板构造企业层静态辅助表。
    包含：firm_id, D, X, Y_pre_mean, Y_post_mean, DeltaY
    """
    pre = panel[panel["post"] == 0].groupby("firm_id")["Y"].mean().rename("Y_pre_mean")
    post = panel[panel["post"] == 1].groupby("firm_id")["Y"].mean().rename("Y_post_mean")
    firm = panel.groupby("firm_id").agg(D=("D", "first"), X=("X", "first")).reset_index()
    firm = firm.merge(pre, on="firm_id", how="left").merge(post, on="firm_id", how="left")
    firm["DeltaY"] = firm["Y_post_mean"] - firm["Y_pre_mean"]
    return firm


def _safe_bias(estimate: float | None, true_effect: float | None) -> float | None:
    """在真值存在时计算 bias。"""
    if estimate is None or true_effect is None or np.isnan(estimate) or np.isnan(true_effect):
        return None
    return float(estimate - true_effect)


def _safe_sq_error(estimate: float | None, true_effect: float | None) -> float | None:
    """在真值存在时计算平方误差。"""
    bias = _safe_bias(estimate, true_effect)
    if bias is None:
        return None
    return float(bias ** 2)


def _tag_match_result(
    match: MatchResult,
    *,
    method: str,
    scenario: str,
    sim_id: int,
) -> MatchResult:
    """为匹配诊断结果补充方法和模拟标识。"""
    return replace(match, method=method, scenario=scenario, sim_id=sim_id)


def _twfe_coef(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str = "Y",
    entity_col: str = "firm_id",
    time_col: str = "year",
) -> float:
    """
    计算双向固定效应（TWFE）下 treatment_col 的系数。

    使用 within-transformation:
        y_it - y_i. - y_.t + y_..
    """
    y = df[outcome_col].values.astype(float)
    treat = df[treatment_col].values.astype(float)

    entity_means_y = df.groupby(entity_col)[outcome_col].transform("mean").values
    entity_means_t = df.groupby(entity_col)[treatment_col].transform("mean").values.astype(float)
    time_means_y = df.groupby(time_col)[outcome_col].transform("mean").values
    time_means_t = df.groupby(time_col)[treatment_col].transform("mean").values.astype(float)

    y_dm = y - entity_means_y - time_means_y + y.mean()
    t_dm = treat - entity_means_t - time_means_t + treat.mean()

    denom = float(np.dot(t_dm, t_dm))
    if abs(denom) < 1e-12:
        return np.nan
    return float(np.dot(t_dm, y_dm) / denom)


def _twfe_multi_coef(
    df: pd.DataFrame,
    treatment_cols: list[str],
    outcome_col: str = "Y",
    entity_col: str = "firm_id",
    time_col: str = "year",
) -> np.ndarray:
    """
    多变量 TWFE within-transformation 求解器。

    对每个 treatment column 做组内去均值，在 demeaned 数据上求解多元 OLS：
        β = (X_dm' X_dm)^{-1} X_dm' y_dm

    返回与 treatment_cols 顺序一致的系数数组。
    若 X_dm' X_dm 不可逆，返回全 NaN 数组。
    """
    y = df[outcome_col].values.astype(float)
    n_vars = len(treatment_cols)

    # 预计算 group means
    entity_means_y = df.groupby(entity_col)[outcome_col].transform("mean").values
    time_means_y = df.groupby(time_col)[outcome_col].transform("mean").values
    grand_y = float(y.mean())

    y_dm = y - entity_means_y - time_means_y + grand_y

    X_dm = np.empty((len(df), n_vars), dtype=float)
    for j, col in enumerate(treatment_cols):
        x = df[col].values.astype(float)
        entity_means_x = df.groupby(entity_col)[col].transform("mean").values.astype(float)
        time_means_x = df.groupby(time_col)[col].transform("mean").values.astype(float)
        grand_x = float(x.mean())
        X_dm[:, j] = x - entity_means_x - time_means_x + grand_x

    XtX = X_dm.T @ X_dm
    Xty = X_dm.T @ y_dm

    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return np.full(n_vars, np.nan)

    return beta


def estimate_event_study(
    panel: pd.DataFrame,
    ref_year: int = 3,
) -> dict[int, float]:
    """
    标准事件研究回归。

    回归形式：
        Y_it = α_i + λ_t + Σ_{k≠ref} β_k × (D_i × I(year==k)) + ε_it

    以 ref_year 为基期（归一化到 0）。返回 {year: coefficient} 映射，
    其中 ref_year 的系数固定为 0.0。

    在含企业 FE 和年份 FE 的面板中，逐期交乘项系数的变化直接反映
    处理组与控制组在各期相对于基期的差异变化。
    """
    df = panel.copy()
    years = sorted(int(y) for y in df["year"].unique())

    if ref_year not in years:
        raise ValueError(f"ref_year {ref_year} not in panel years {years}")

    treatment_cols: list[str] = []
    year_map: list[int] = []

    for y in years:
        if y == ref_year:
            continue
        col = f"_D_yr_{y}"
        df[col] = (df["D"].astype(float) * (df["year"] == y).astype(float)).values
        treatment_cols.append(col)
        year_map.append(y)

    betas = _twfe_multi_coef(df, treatment_cols)

    result: dict[int, float] = {}
    beta_idx = 0
    for y in years:
        if y == ref_year:
            result[y] = 0.0
        else:
            result[y] = float(betas[beta_idx])
            beta_idx += 1

    return result


def _build_matched_panel(panel: pd.DataFrame, match: MatchResult) -> pd.DataFrame:
    """
    根据匹配结果构造匹配后的长面板样本。

    处理组企业保留一次；控制组企业若被重复匹配，则按匹配次数复制，
    并重命名 firm_id，以便在固定效应回归中等价体现权重。

    保留 original_firm_id 列，供按原始企业聚类等操作使用。
    """
    treated_firm_ids = set(match.treated_ids)
    control_firm_ids = list(match.control_ids)

    matched_panel = panel[panel["firm_id"].isin(treated_firm_ids)].copy()
    matched_panel["original_firm_id"] = matched_panel["firm_id"]
    ctrl_panel = panel[panel["firm_id"].isin(control_firm_ids)].copy()

    ctrl_firm_counts = pd.Series(control_firm_ids).value_counts()
    expanded_ctrl_parts = []
    for fid, count in ctrl_firm_counts.items():
        firm_rows = ctrl_panel[ctrl_panel["firm_id"] == fid].copy()
        for rep in range(count):
            rep_rows = firm_rows.copy()
            rep_rows["firm_id"] = f"{fid}_rep{rep}"
            rep_rows["original_firm_id"] = fid
            expanded_ctrl_parts.append(rep_rows)

    if expanded_ctrl_parts:
        expanded_ctrl = pd.concat(expanded_ctrl_parts, ignore_index=True)
        matched_panel = pd.concat([matched_panel, expanded_ctrl], ignore_index=True)

    return matched_panel


def _compute_matched_att_true(
    firm_level: pd.DataFrame,
    match: MatchResult,
) -> float:
    """
    计算给定匹配结果下、匹配后处理组样本对应的真实 ATT。

    在当前 1:1 放回匹配实现中，每个处理组企业恰好出现一次，因此该值等于
    匹配进入样本的 treated firm 的平均 τ_i。
    """
    treated_tau = (
        firm_level.loc[firm_level["firm_id"].isin(match.treated_ids), ["firm_id", "tau_i"]]
        .drop_duplicates(subset=["firm_id"])
        .set_index("firm_id")["tau_i"]
    )
    matched_tau = pd.Series(match.treated_ids).map(treated_tau)
    return float(matched_tau.mean())


def prepare_matching(
    panel: pd.DataFrame,
    scenario: str,
    sim_id: int,
) -> MatchingPrep:
    """
    为 PSM 与 PSM-DID 统一准备 firm-level 数据、倾向得分、共同支持裁剪与匹配结果。
    """
    firm = _build_firm_level(panel)
    if "tau_i" not in firm.columns:
        if scenario == "D":
            firm["tau_i"] = heterogeneous_tau_from_x(firm["X"].values.astype(float))
        else:
            firm["tau_i"] = TAU

    truth = compute_true_effect_stats(
        scenario,
        sim_id,
        firm["X"].values.astype(float),
        firm["D"].values.astype(int),
    )
    truth_stats: dict[str, float | int] = {
        "ate_true": truth.ate_true,
        "att_true": truth.att_true,
        "atc_true": truth.atc_true,
        "tau_sd_true": truth.tau_sd_true,
        "tau_min_true": truth.tau_min_true,
        "tau_max_true": truth.tau_max_true,
        "n_treated": truth.n_treated,
        "n_control": truth.n_control,
    }
    propensity = _estimate_propensity(firm)
    trimmed_firm, trimmed_p, cs_diag = _common_support_trim(firm, propensity)
    if cs_diag.common_support_failure:
        return MatchingPrep(
            scenario=scenario,
            sim_id=sim_id,
            firm_level=firm,
            propensity=propensity,
            trimmed_firm=trimmed_firm,
            trimmed_propensity=trimmed_p,
            common_support_diag=cs_diag,
            match=None,
            true_effect_stats=truth_stats,
        )

    match = _nearest_neighbor_match(trimmed_firm, trimmed_p, method="shared")
    return MatchingPrep(
        scenario=scenario,
        sim_id=sim_id,
        firm_level=firm,
        propensity=propensity,
        trimmed_firm=trimmed_firm,
        trimmed_propensity=trimmed_p,
        common_support_diag=cs_diag,
        match=match,
        true_effect_stats=truth_stats,
    )


def _estimate_propensity(firm_level: pd.DataFrame) -> np.ndarray:
    """
    用 Logit 模型估计倾向得分 P(D=1 | X)。
    返回与 firm_level 对应的倾向得分数组。
    """
    X_design = sm.add_constant(firm_level["X"].values)
    y = firm_level["D"].values
    try:
        model = sm.Logit(y, X_design)
        result = model.fit(disp=0)
        p = result.predict(X_design)
    except Exception:
        # 如果 Logit 不收敛，回退到均匀先验
        p = np.full(len(firm_level), y.mean())
    return p


def _common_support_trim(
    firm_level: pd.DataFrame,
    propensity: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, MatchResult]:
    """
    执行共同支持区间裁剪。
    返回裁剪后的 firm_level、propensity 和诊断信息。
    """
    treated_mask = firm_level["D"].values == 1
    control_mask = firm_level["D"].values == 0

    p_treated = propensity[treated_mask]
    p_control = propensity[control_mask]

    n_before_t = treated_mask.sum()
    n_before_c = control_mask.sum()

    cs_min = max(p_treated.min(), p_control.min())
    cs_max = min(p_treated.max(), p_control.max())
    cs_width = cs_max - cs_min

    if cs_min >= cs_max:
        # 共同支持区间为空
        diag = MatchResult(
            method="",  # 裁剪失败时尚未区分方法
            treated_ids=np.array([]),
            control_ids=np.array([]),
            match_distances=np.array([]),
            n_treated=0,
            n_control_unique=0,
            common_support_min=cs_min,
            common_support_max=cs_max,
            common_support_width=cs_width,
            common_support_failure=True,
            narrow_support_warning=cs_width > 0 and cs_width < 0.01,
            n_before_trim_treated=n_before_t,
            n_before_trim_control=n_before_c,
            n_after_trim_treated=0,
            n_after_trim_control=0,
        )
        return pd.DataFrame(), np.array([]), diag

    # 裁剪：保留在共同支持区间内的样本
    keep = (propensity >= cs_min) & (propensity <= cs_max)
    trimmed_firm = firm_level.loc[keep].copy()
    trimmed_p = propensity[keep]

    n_after_t = (trimmed_firm["D"].values == 1).sum()
    n_after_c = (trimmed_firm["D"].values == 0).sum()

    diag = MatchResult(
        method="",  # 裁剪阶段尚未区分方法
        treated_ids=np.array([]),
        control_ids=np.array([]),
        match_distances=np.array([]),
        n_treated=0,
        n_control_unique=0,
        common_support_min=cs_min,
        common_support_max=cs_max,
        common_support_width=cs_width,
        common_support_failure=False,
        narrow_support_warning=cs_width > 0 and cs_width < 0.01,
        n_before_trim_treated=n_before_t,
        n_before_trim_control=n_before_c,
        n_after_trim_treated=n_after_t,
        n_after_trim_control=n_after_c,
    )
    return trimmed_firm, trimmed_p, diag


def _nearest_neighbor_match(
    firm_level: pd.DataFrame,
    propensity: np.ndarray,
    method: str,  # "PSM" or "PSM-DID"
) -> MatchResult:
    """
    1:1 最近邻匹配（放回）。
    传入的 firm_level 和 propensity 应已完成共同支持裁剪。
    """
    treated_mask = firm_level["D"].values == 1
    control_mask = firm_level["D"].values == 0

    treated_idx = np.where(treated_mask)[0]
    control_idx = np.where(control_mask)[0]

    p_treated = propensity[treated_idx]
    p_control = propensity[control_idx]

    treated_firm_ids = firm_level.iloc[treated_idx]["firm_id"].values
    control_firm_ids = firm_level.iloc[control_idx]["firm_id"].values

    # 为每个处理组企业找到最近的控制组企业（O(T×log(C)) 代替 O(T×C)）
    sorted_indices = np.argsort(p_control)
    p_control_sorted = p_control[sorted_indices]

    match_ctrl_indices = []
    match_distances = []
    for p_t in p_treated:
        idx = np.searchsorted(p_control_sorted, p_t)
        n = len(p_control_sorted)
        if idx == 0:
            best_idx = 0
        elif idx == n:
            best_idx = n - 1
        else:
            left = p_control_sorted[idx - 1]
            right = p_control_sorted[idx]
            best_idx = idx if (p_t - left) >= (right - p_t) else idx - 1
        match_ctrl_indices.append(best_idx)
        match_distances.append(abs(p_control_sorted[best_idx] - p_t))

    match_ctrl_indices = np.array(match_ctrl_indices)
    match_distances = np.array(match_distances)

    # 将排序后的索引映射回原始 control_idx
    match_ctrl_original = sorted_indices[match_ctrl_indices]

    # 将排序后的索引映射回原始 control_idx，再用 control_firm_ids 取真实 firm_id
    matched_control_firm_ids = control_firm_ids[match_ctrl_original]
    n_control_unique = len(np.unique(matched_control_firm_ids))

    # 共同支持区间
    cs_min = max(p_treated.min(), p_control.min()) if len(p_treated) > 0 and len(p_control) > 0 else 0.0
    cs_max = min(p_treated.max(), p_control.max()) if len(p_treated) > 0 and len(p_control) > 0 else 0.0

    return MatchResult(
        method=method,
        treated_ids=treated_firm_ids,
        control_ids=matched_control_firm_ids,
        match_distances=match_distances,
        n_treated=len(treated_firm_ids),
        n_control_unique=n_control_unique,
        common_support_min=cs_min,
        common_support_max=cs_max,
        common_support_width=cs_max - cs_min,
        common_support_failure=False,
        narrow_support_warning=(cs_max - cs_min) > 0 and (cs_max - cs_min) < 0.01,
        n_before_trim_treated=len(treated_idx),
        n_before_trim_control=len(control_idx),
        n_after_trim_treated=len(treated_idx),
        n_after_trim_control=len(control_idx),
    )


# ---------------------------------------------------------------------------
# 三种方法
# ---------------------------------------------------------------------------

def estimate_psm(
    panel: pd.DataFrame,
    scenario: str,
    sim_id: int,
    true_tau: float | None = 50.0,
    prep: MatchingPrep | None = None,
) -> tuple[EstimationResult, MatchResult | None]:
    """
    单独 PSM：1:1 最近邻匹配 + 放回 + 共同支持裁剪。
    估计对象：匹配后的 ATT（基于处理后结果水平）。
    """
    if prep is None:
        prep = prepare_matching(panel, scenario, sim_id)

    trimmed_firm = prep.trimmed_firm
    cs_diag = prep.common_support_diag
    if cs_diag.common_support_failure:
        target_true = float(prep.true_effect_stats["att_true"])
        cs_diag = _tag_match_result(cs_diag, method="PSM", scenario=scenario, sim_id=sim_id)
        result = EstimationResult(
            scenario=scenario, sim_id=sim_id, method="PSM",
            estimate=None, bias=None, sq_error=None,
            status="common_support_failure",
            common_support_width=cs_diag.common_support_width,
            n_before_trim=cs_diag.n_before_trim_treated + cs_diag.n_before_trim_control,
            n_after_trim=0,
            target_estimand="ATT",
            true_effect_target=target_true,
        )
        return result, cs_diag

    # 匹配
    if prep.match is None:
        raise ValueError("Matching preparation unexpectedly missing match result.")
    match = _tag_match_result(prep.match, method="PSM", scenario=scenario, sim_id=sim_id)

    # 计算 PSM 估计：匹配后比较处理后结果水平（而非结果变化）
    treated_post = trimmed_firm.loc[
        trimmed_firm["firm_id"].isin(match.treated_ids), ["firm_id", "Y_post_mean"]
    ].set_index("firm_id")["Y_post_mean"]

    # 构建匹配对：处理组 firm_id → 控制组 firm_id
    match_df = pd.DataFrame({
        "treated_firm_id": match.treated_ids,
        "control_firm_id": match.control_ids,
    })
    ctrl_post = trimmed_firm.loc[
        trimmed_firm["firm_id"].isin(match.control_ids), ["firm_id", "Y_post_mean"]
    ].set_index("firm_id")["Y_post_mean"]

    # 按匹配对计算
    match_df["treated_post"] = match_df["treated_firm_id"].map(treated_post).values
    match_df["control_post"] = match_df["control_firm_id"].map(ctrl_post).values
    tau_hat = (match_df["treated_post"] - match_df["control_post"]).mean()
    target_true = _compute_matched_att_true(trimmed_firm, match)

    result = EstimationResult(
        scenario=scenario, sim_id=sim_id, method="PSM",
        estimate=tau_hat,
        bias=_safe_bias(tau_hat, target_true),
        sq_error=_safe_sq_error(tau_hat, target_true),
        status="ok",
        n_matched_pairs=match.n_treated,
        n_control_unique=match.n_control_unique,
        avg_match_distance=match.match_distances.mean() if len(match.match_distances) > 0 else None,
        common_support_width=match.common_support_width,
        n_before_trim=cs_diag.n_before_trim_treated + cs_diag.n_before_trim_control,
        n_after_trim=cs_diag.n_after_trim_treated + cs_diag.n_after_trim_control,
        target_estimand="ATT",
        true_effect_target=target_true,
    )
    return result, match


def estimate_did(
    panel: pd.DataFrame,
    scenario: str,
    sim_id: int,
    true_tau: float | None = 50.0,
) -> EstimationResult:
    """
    单独 DID：TWFE 面板回归，不做匹配。
    Y_it = alpha_i + lambda_t + theta * (D_i × post_t) + epsilon_it

    使用双重去均值（within-transformation）代替 OLS 虚拟变量，等价但快得多。
    """
    df = panel.copy()
    df["D_post"] = df["D"] * df["post"]
    tau_hat = _twfe_coef(df, treatment_col="D_post")
    truth = compute_true_effect_stats(
        scenario,
        sim_id,
        df.groupby("firm_id")["X"].first().values.astype(float),
        df.groupby("firm_id")["D"].first().values.astype(int),
    )
    target_true = truth.att_true

    return EstimationResult(
        scenario=scenario, sim_id=sim_id, method="DID",
        estimate=tau_hat,
        bias=_safe_bias(tau_hat, target_true),
        sq_error=_safe_sq_error(tau_hat, target_true),
        status="ok",
        target_estimand="ATT",
        true_effect_target=target_true,
    )


def estimate_did_x(
    panel: pd.DataFrame,
    scenario: str,
    sim_id: int,
    true_tau: float | None = 50.0,
) -> EstimationResult:
    """
    控制协变量 × 时间趋势的 DID（"DID-X"）：
        Y_it = alpha_i + lambda_t + theta * (D_i × post_t) + gamma * (X_i × s_t) + epsilon_it

    其中 s_t = t - 1（线性时间刻度，t ∈ {1,2,3,4,5,6}）。

    X_i 为时间不变协变量，直接加入 X_i 水平项会被企业固定效应吸收；
    加入 X_i × s_t 后变为时变项（X_i × 0, X_i × 1, ..., X_i × 5），
    可以捕捉由可观测协变量驱动的线性趋势异质性（场景 B）。

    回归使用 within-transformation（去企业均值 + 去时间均值 + 加回总均值），
    在 demeaned 数据上做 OLS 回归 y_dm ~ d_dm + x_s_dm，提取 theta。

    Parameters
    ----------
    panel : pd.DataFrame
        原始 6 期面板。
    scenario, sim_id, true_tau : 标识与真值。

    Returns
    -------
    EstimationResult
        method = "DID-X"，其余字段同 estimate_did。
    """
    df = panel.copy()
    df["D_post"] = df["D"] * df["post"]

    # s_t = t - 1：线性时间刻度（与任务一定义一致）
    df["s_t"] = df["year"] - 1
    # X_i × s_t：时变交互项
    df["X_s"] = df["X"] * df["s_t"]

    # within-transformation
    y = df["Y"].values.astype(float)
    dpost = df["D_post"].values.astype(float)
    x_s = df["X_s"].values.astype(float)

    firm_means_y = df.groupby("firm_id")["Y"].transform("mean").values
    firm_means_d = df.groupby("firm_id")["D_post"].transform("mean").values.astype(float)
    firm_means_xs = df.groupby("firm_id")["X_s"].transform("mean").values.astype(float)

    year_means_y = df.groupby("year")["Y"].transform("mean").values
    year_means_d = df.groupby("year")["D_post"].transform("mean").values.astype(float)
    year_means_xs = df.groupby("year")["X_s"].transform("mean").values.astype(float)

    grand_mean_y = y.mean()
    grand_mean_d = dpost.mean()
    grand_mean_xs = x_s.mean()

    y_dm = y - firm_means_y - year_means_y + grand_mean_y
    d_dm = dpost - firm_means_d - year_means_d + grand_mean_d
    x_s_dm = x_s - firm_means_xs - year_means_xs + grand_mean_xs

    # OLS: y_dm ~ d_dm + x_s_dm（无截距，within 已吸收均值）
    # [d_dm, x_s_dm]' [d_dm, x_s_dm] * [theta, gamma]' = [d_dm, x_s_dm]' y_dm
    dd = np.dot(d_dm, d_dm)
    dx = np.dot(d_dm, x_s_dm)
    xd = dx  # symmetric
    xx = np.dot(x_s_dm, x_s_dm)
    dy = np.dot(d_dm, y_dm)
    xy = np.dot(x_s_dm, y_dm)

    det = dd * xx - dx * xd
    if abs(det) < 1e-12:
        tau_hat = np.nan
    else:
        # [theta, gamma]' = A^{-1} * b
        theta = (dy * xx - dx * xy) / det
        gamma = (dd * xy - xd * dy) / det
        tau_hat = theta

    truth = compute_true_effect_stats(
        scenario,
        sim_id,
        df.groupby("firm_id")["X"].first().values.astype(float),
        df.groupby("firm_id")["D"].first().values.astype(int),
    )
    target_true = truth.att_true

    return EstimationResult(
        scenario=scenario, sim_id=sim_id, method="DID-X",
        estimate=tau_hat,
        bias=_safe_bias(tau_hat, target_true) if not np.isnan(tau_hat) else None,
        sq_error=_safe_sq_error(tau_hat, target_true) if not np.isnan(tau_hat) else None,
        status="ok" if not np.isnan(tau_hat) else "numerical_failure",
        target_estimand="ATT",
        true_effect_target=target_true,
    )


def estimate_psm_did(
    panel: pd.DataFrame,
    scenario: str,
    sim_id: int,
    true_tau: float | None = 50.0,
    prep: MatchingPrep | None = None,
) -> tuple[EstimationResult, MatchResult | None]:
    """
    PSM-DID：先匹配，再在匹配后面板上做 DID 回归。
    """
    if prep is None:
        prep = prepare_matching(panel, scenario, sim_id)

    cs_diag = prep.common_support_diag
    if cs_diag.common_support_failure:
        target_true = float(prep.true_effect_stats["att_true"])
        cs_diag = _tag_match_result(cs_diag, method="PSM-DID", scenario=scenario, sim_id=sim_id)
        result = EstimationResult(
            scenario=scenario, sim_id=sim_id, method="PSM-DID",
            estimate=None, bias=None, sq_error=None,
            status="common_support_failure",
            common_support_width=cs_diag.common_support_width,
            n_before_trim=cs_diag.n_before_trim_treated + cs_diag.n_before_trim_control,
            n_after_trim=0,
            target_estimand="ATT",
            true_effect_target=target_true,
        )
        return result, cs_diag

    # 匹配
    if prep.match is None:
        raise ValueError("Matching preparation unexpectedly missing match result.")
    match = _tag_match_result(prep.match, method="PSM-DID", scenario=scenario, sim_id=sim_id)

    # 构建匹配后的面板样本
    matched_panel = _build_matched_panel(panel, match)
    if matched_panel.empty or not (matched_panel["D"] == 0).any():
        target_true = _compute_matched_att_true(prep.trimmed_firm, match)
        # 没有匹配到控制组
        result = EstimationResult(
            scenario=scenario, sim_id=sim_id, method="PSM-DID",
            estimate=None, bias=None, sq_error=None,
            status="common_support_failure",
            common_support_width=match.common_support_width,
            n_before_trim=cs_diag.n_before_trim_treated + cs_diag.n_before_trim_control,
            n_after_trim=cs_diag.n_after_trim_treated + cs_diag.n_after_trim_control,
            target_estimand="ATT",
            true_effect_target=target_true,
        )
        return result, match

    # DID 回归（within-transformation，同 estimate_did）
    matched_panel["D_post"] = matched_panel["D"] * matched_panel["post"]
    tau_hat = _twfe_coef(matched_panel, treatment_col="D_post")
    target_true = _compute_matched_att_true(prep.trimmed_firm, match)

    result = EstimationResult(
        scenario=scenario, sim_id=sim_id, method="PSM-DID",
        estimate=tau_hat,
        bias=_safe_bias(tau_hat, target_true),
        sq_error=_safe_sq_error(tau_hat, target_true),
        status="ok",
        n_matched_pairs=match.n_treated,
        n_control_unique=match.n_control_unique,
        avg_match_distance=match.match_distances.mean() if len(match.match_distances) > 0 else None,
        common_support_width=match.common_support_width,
        n_before_trim=cs_diag.n_before_trim_treated + cs_diag.n_before_trim_control,
        n_after_trim=cs_diag.n_after_trim_treated + cs_diag.n_after_trim_control,
        target_estimand="ATT",
        true_effect_target=target_true,
    )
    return result, match


# ---------------------------------------------------------------------------
# 单次模拟的完整估计流程
# ---------------------------------------------------------------------------

def run_one_simulation(
    panel: pd.DataFrame,
    scenario: str,
    sim_id: int,
    true_tau: float = 50.0,
) -> tuple[list[EstimationResult], list[MatchResult], MatchingPrep]:
    """
    对单次模拟数据执行三种方法的估计。
    返回估计结果列表、匹配诊断列表和共享匹配准备结果。
    """
    estimates = []
    matches = []
    prep = prepare_matching(panel, scenario, sim_id)

    # 1. 基础 DID（无协变量控制）
    did_res = estimate_did(panel, scenario, sim_id, true_tau)
    estimates.append(did_res)

    # 2. DID-X（控制处理前协变量 X）
    didx_res = estimate_did_x(panel, scenario, sim_id, true_tau)
    estimates.append(didx_res)

    # 3. PSM
    psm_res, psm_match = estimate_psm(panel, scenario, sim_id, true_tau, prep=prep)
    estimates.append(psm_res)
    if psm_match is not None:
        matches.append(psm_match)

    # 4. PSM-DID
    psmdid_res, psmdid_match = estimate_psm_did(panel, scenario, sim_id, true_tau, prep=prep)
    estimates.append(psmdid_res)
    if psmdid_match is not None:
        matches.append(psmdid_match)

    return estimates, matches, prep
