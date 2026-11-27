"""Capture the software/hardware environment, and shared helpers for every benchmark.

Run directly to print the environment and store it:
    python -m bench.run_env

Every benchmark result JSON embeds the output of `environment()`, so each number in
bench/results/ can be traced to a GPU, driver, CUDA version, library versions and commit.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import statistics
import subprocess
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"
_PACKAGES = ["numpy", "scipy", "cupy-cuda12x", "qiskit", "qiskit-aer", "qiskit-aer-gpu",
             "cuquantum-python-cu12", "qsim-gpu"]


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_commit() -> str | None:
    root = Path(__file__).resolve().parent.parent
    commit = _run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"])
    # Ignore bench/results: a benchmark run rewrites those files itself, which is not a code change.
    dirty = _run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no", "--",
                  ".", ":(exclude)bench/results"])
    if commit and dirty:
        commit += "-dirty"
    return commit


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,clocks.max.sm,"
                "clocks.max.mem", "--format=csv,noheader"])
    if smi:
        name, mem, driver, sm_clock, mem_clock = [s.strip() for s in smi.splitlines()[0].split(",")]
        info.update(nvidia_smi_name=name, memory_total=mem, driver=driver,
                    max_sm_clock=sm_clock, max_mem_clock=mem_clock)
    state = _run(["nvidia-smi", "--query-gpu=pstate,clocks.sm,clocks.mem,power.draw,"
                  "temperature.gpu,clocks_event_reasons.active,pcie.link.gen.current",
                  "--format=csv,noheader"])
    if state:
        keys = ["pstate", "current_sm_clock", "current_mem_clock", "power_draw",
                "temperature_c", "clock_event_reasons", "pcie_gen"]
        info["state_at_run"] = dict(zip(keys, [v.strip() for v in state.splitlines()[0].split(",")],
                                        strict=False))
        if "not in ready state" in state:
            info["warning"] = ("driver reports 'System is not in ready state' for clocks; "
                               "bandwidth numbers from this run are not representative")
    try:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        info.update(
            name=props["name"].decode(),
            compute_capability=f"{props['major']}.{props['minor']}",
            total_memory_bytes=int(props["totalGlobalMem"]),
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            cuda_driver_api=cp.cuda.runtime.driverGetVersion(),
            nvrtc=".".join(map(str, cp.cuda.nvrtc.getVersion())),
        )
    except Exception as exc:  # no GPU or no CuPy: record why
        info["cupy_error"] = f"{type(exc).__name__}: {exc}"
    return info


def environment() -> dict[str, Any]:
    """Everything needed to reproduce or interpret a benchmark number."""
    versions: dict[str, str | None] = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "git_commit": _git_commit(),
        "gpu": _gpu_info(),
        "nvcc": (_run(["nvcc", "--version"]) or "not installed").splitlines()[-1],
        "packages": versions,
    }


def time_gpu(fn: Callable[[], object], repeats: int = 7, warmup: int = 2) -> dict[str, Any]:
    """Time `fn` with CUDA events. Returns median/min/max in milliseconds over `repeats` runs."""
    import cupy as cp

    for _ in range(warmup):
        fn()
    cp.cuda.Device().synchronize()
    times = []
    for _ in range(repeats):
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop))
    return {"median_ms": statistics.median(times), "min_ms": min(times),
            "max_ms": max(times), "repeats": repeats, "all_ms": times}


def save_result(name: str, payload: dict[str, Any]) -> Path:
    """Write payload + environment to bench/results/<name>.json and return the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps({"benchmark": name, "environment": environment(), **payload},
                               indent=2) + "\n")
    return path


if __name__ == "__main__":
    env = environment()
    print(json.dumps(env, indent=2))
    gpu = str(env["gpu"].get("name", "cpu-only"))
    slug = "".join(c if c.isalnum() else "-" for c in gpu.lower()).strip("-")
    out = save_result(f"env_{slug}", {})
    print(f"saved {out}")
