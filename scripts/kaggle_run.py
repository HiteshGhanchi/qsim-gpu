"""Run the GPU tests and benchmarks on a free Kaggle T4 and download the results.

Needs the Kaggle CLI and your own credentials in ~/.kaggle/ (this script never reads them; the
CLI does). Phone verification on Kaggle is required for GPUs.

    python scripts/kaggle_run.py push      # upload and start a run at the current commit
    python scripts/kaggle_run.py status
    python scripts/kaggle_run.py fetch     # download results.zip into kaggle_out/

The commit must already be on GitHub, because the Kaggle machine clones it from there.
The kernel is private and named <your-username>/qsim-gpu-benchmarks.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = os.environ.get("KAGGLE_CLI", "kaggle")
SLUG = "qsim-gpu-benchmarks"


def kaggle(*args: str) -> str:
    return subprocess.run([KAGGLE, *args], check=True, capture_output=True, text=True).stdout


def username() -> str:
    match = re.search(r"username:\s*(\S+)", kaggle("config", "view"))
    if not match:
        sys.exit("could not find your Kaggle username; is the Kaggle CLI configured?")
    return match.group(1)


def push() -> None:
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        script = (ROOT / "notebooks/kaggle/run_on_kaggle.py").read_text()
        script = script.replace('REF = "main"', f'REF = "{commit}"')
        Path(tmp, "run_on_kaggle.py").write_text(script)
        Path(tmp, "kernel-metadata.json").write_text(json.dumps({
            "id": f"{username()}/{SLUG}", "title": SLUG, "code_file": "run_on_kaggle.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "dataset_sources": [],
            "competition_sources": [], "kernel_sources": []}, indent=2))
        print(kaggle("kernels", "push", "-p", tmp, "--accelerator", "NvidiaTeslaT4"))
    print(f"started run at commit {commit[:7]}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    ref = f"{username()}/{SLUG}"
    if cmd == "push":
        push()
    elif cmd == "status":
        print(kaggle("kernels", "status", ref))
    elif cmd == "fetch":
        out = ROOT / "kaggle_out"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        print(kaggle("kernels", "output", ref, "-p", str(out)))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
