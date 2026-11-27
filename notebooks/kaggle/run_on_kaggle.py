"""Kaggle script: clone qsim-gpu, run every test and benchmark on the T4, keep results.zip.

Pushed to Kaggle by `python scripts/kaggle_run.py`; Kaggle keeps /kaggle/working as output.
"""

import os
import shutil
import subprocess

REPO = "https://github.com/HiteshGhanchi/qsim-gpu.git"
REF = "main"  # replaced with an exact commit by scripts/kaggle_run.py
TESTS_ONLY = "0"  # "1" = run the test suite only (set by kaggle_run.py push --tests-only)

subprocess.run(["git", "clone", "-q", REPO, "qsim-gpu"], check=True)
subprocess.run(["git", "-C", "qsim-gpu", "checkout", "-q", REF], check=True)
env = {**os.environ, "QSIM_TESTS_ONLY": TESTS_ONLY}
subprocess.run(["bash", "scripts/gpu_cloud_run.sh"], cwd="qsim-gpu", check=False, env=env)
shutil.copy("qsim-gpu/results.zip", "/kaggle/working/results.zip")
