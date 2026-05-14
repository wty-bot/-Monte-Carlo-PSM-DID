"""
路径管理工具
"""
from __future__ import annotations

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# data/ 子目录
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"

# outputs/ 子目录
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_CHECKS = PROJECT_ROOT / "outputs" / "checks"
OUTPUT_LOGS = PROJECT_ROOT / "outputs" / "logs"

# results/（兼容旧目录）
RESULTS_DIR = PROJECT_ROOT / "results"


def ensure_dirs() -> None:
    """确保所有输出目录存在（静默创建）"""
    for d in [
        DATA_RAW,
        DATA_INTERIM,
        DATA_PROCESSED,
        DATA_EXTERNAL,
        OUTPUT_FIGURES,
        OUTPUT_TABLES,
        OUTPUT_CHECKS,
        OUTPUT_LOGS,
        RESULTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def raw_path(scenario: str, kind: str = "final") -> Path:
    """
    返回 data/raw/scenario_X/ 下的文件路径。

    Parameters
    ----------
    scenario : str
        场景标识，如 "A"、"B"、"C"
    kind : str
        "final" → 正式交付数据（不含 U_i / Y(0) / Y(1)）
        "debug" → 调试版本（含真值字段）

    Returns
    -------
    Path
    """
    ensure_dirs()
    subdir = DATA_RAW / f"scenario_{scenario.lower()}"
    subdir.mkdir(parents=True, exist_ok=True)

    if kind == "debug":
        return subdir / f"task1_debug_scenario_{scenario.lower()}.parquet"
    return subdir / f"task1_scenario_{scenario.lower()}.parquet"