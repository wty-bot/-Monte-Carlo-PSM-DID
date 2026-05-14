"""
随机种子管理工具

每个场景的种子策略（来自任务一规划文档）：
- 场景 A：seed = 1000 + sim_id
- 场景 B：seed = 2000 + sim_id
- 场景 C：seed = 3000 + sim_id
- 场景 D：seed = 4000 + sim_id

每次 sim_id 循环内部创建独立 RNG，不依赖全局随机状态。
"""
from __future__ import annotations

import numpy as np


def make_rng(scenario: str, sim_id: int) -> np.random.Generator:
    """
    为指定场景和模拟编号创建独立的随机数生成器。

    Parameters
    ----------
    scenario : str
        场景标识，"A"、"B"、"C" 或 "D"
    sim_id : int
        模拟编号，从 1 开始

    Returns
    -------
    np.random.Generator
    """
    base = {"A": 1000, "B": 2000, "C": 3000, "D": 4000}[scenario]
    return np.random.default_rng(base + sim_id)

SCENARIO_SEEDS = {"A": 1000, "B": 2000, "C": 3000, "D": 4000}
"""
各场景的种子基准值。
通过 seed = SCENARIO_SEEDS[scenario] + sim_id 计算每次循环的实际种子。
"""
