#!/usr/bin/env python
"""Print the exact execution environment behind a benchmark run.

The problem statement requires runtime to be reported together with the
hardware, Python version and timing method it was measured on. Runtime is the
one deliverable figure that is genuinely machine-dependent - the same frozen
156-pair benchmark takes 3.72 s/pair on the development workstation and
6.08 s/pair on a 2-core container - so a bare number is not reportable on its
own.

This prints a block that can be pasted directly into the README or the
solution PPT, read live from the machine it runs on rather than transcribed by
hand.

    python scripts/report_environment.py
"""
from __future__ import annotations

import os
import platform
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except ImportError:
        return "not installed"


def main() -> None:
    print("Execution environment")
    print("=" * 62)

    rows = [
        ("OS", f"{platform.system()} {platform.release()} ({platform.version()})"),
        ("Machine / arch", f"{platform.machine()} / {platform.architecture()[0]}"),
        ("CPU", platform.processor() or "unknown"),
        ("Logical cores", str(os.cpu_count())),
        ("Python", f"{platform.python_version()} ({platform.python_implementation()})"),
    ]
    for name in ("cv2", "numpy", "scipy", "pandas", "matplotlib", "streamlit"):
        rows.append((name, _version(name)))

    try:
        import torch
        rows.append(("torch", torch.__version__))
        rows.append(("CUDA available", str(torch.cuda.is_available())))
    except ImportError:
        rows.append(("torch", "not installed (not needed for the production path)"))

    for label, value in rows:
        print(f"  {label:<16} {value}")

    print()
    print("Timing method")
    print("=" * 62)
    print("  time.perf_counter() wrapped around the localize() call only")
    print("  (pipeline/localize.py). Covers candidate generation, scoring,")
    print("  ranking and sub-pixel refinement. EXCLUDES image file I/O and")
    print("  dataset generation. Single process, CPU only, no GPU.")
    print()
    print("  Per-pair runtimes for the last run are the `runtime_s` column of")
    print("  outputs/reports/per_pair_results.csv; the mean and median are")
    print("  `mean_runtime_s` / `median_runtime_s` in baseline_metrics.json.")


if __name__ == "__main__":
    main()
