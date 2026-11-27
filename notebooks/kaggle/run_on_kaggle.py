"""Kaggle script: clone qsim-gpu, run every test and benchmark on the T4, keep results.zip.

Pushed to Kaggle by `python scripts/kaggle_run.py`; Kaggle keeps /kaggle/working as output.
"""

import shutil
import subprocess

REPO = "https://github.com/HiteshGhanchi/qsim-gpu.git"
REF = "main"  # replaced with an exact commit by scripts/kaggle_run.py

subprocess.run(["git", "clone", "-q", REPO, "qsim-gpu"], check=True)
subprocess.run(["git", "-C", "qsim-gpu", "checkout", "-q", REF], check=True)
subprocess.run(["bash", "scripts/gpu_cloud_run.sh"], cwd="qsim-gpu", check=False)
shutil.copy("qsim-gpu/results.zip", "/kaggle/working/results.zip")
