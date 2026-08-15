#!/usr/bin/env python
"""Launch the DriftSense V2 Streamlit application.

    python scripts/run_demo.py [--port 8501]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    app_path = os.path.join(PROJECT_ROOT, "app", "app.py")
    # --server.headless true skips Streamlit's interactive first-run email
    # prompt (which otherwise blocks on stdin the first time it's ever run
    # on a machine) - the local URL below still works in any browser.
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path,
         "--server.port", str(args.port), "--server.headless", "true"],
        cwd=PROJECT_ROOT, check=True,
    )


if __name__ == "__main__":
    main()
