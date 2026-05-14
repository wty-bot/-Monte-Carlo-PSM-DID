"""
utils包：通用工具（读写、路径管理、随机种子等）
"""

from .paths import (
    PROJECT_ROOT,
    DATA_RAW,
    DATA_INTERIM,
    DATA_PROCESSED,
    DATA_EXTERNAL,
    OUTPUT_FIGURES,
    OUTPUT_TABLES,
    OUTPUT_CHECKS,
    OUTPUT_LOGS,
    RESULTS_DIR,
    ensure_dirs,
    raw_path,
)
from .seeds import make_rng, SCENARIO_SEEDS

__all__ = [
    "PROJECT_ROOT",
    "DATA_RAW",
    "DATA_INTERIM",
    "DATA_PROCESSED",
    "DATA_EXTERNAL",
    "OUTPUT_FIGURES",
    "OUTPUT_TABLES",
    "OUTPUT_CHECKS",
    "OUTPUT_LOGS",
    "RESULTS_DIR",
    "ensure_dirs",
    "raw_path",
    "make_rng",
    "SCENARIO_SEEDS",
]