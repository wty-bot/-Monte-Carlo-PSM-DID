from __future__ import annotations

import importlib
import platform
import sys


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "sklearn",
    "pyarrow",
    "matplotlib",
    "seaborn",
    "pytest",
    "dotenv",
]


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    missing: list[str] = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            print(f"OK {module_name}")
        except Exception as exc:  # pragma: no cover - diagnostic script
            print(f"MISSING {module_name}: {exc}")
            missing.append(module_name)

    if missing:
        print("\nEnvironment check failed.")
        print("Missing modules:", ", ".join(missing))
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
