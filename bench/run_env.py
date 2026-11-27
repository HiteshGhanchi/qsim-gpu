"""Capture the software/hardware environment, and shared helpers for every benchmark.

Run directly to print the environment and store it:
    python -m bench.run_env

Every benchmark result JSON embeds the output of `environment()`, so each number in
bench/results/ can be traced to a GPU, driver, CUDA version, library versions and commit.

Results are stored per GPU, in bench/results/<gpu-slug>/ (for example tesla-t4/), so runs on
different machines never overwrite each other.

Bandwidth sanity check: `environment()` also times a 64 MiB copy on the GPU. If the GPU has a
published memory bandwidth in PUBLISHED_BANDWIDTH and the measured copy reaches less than
SUSPECT_FRACTION of it, the result is marked "suspect" and a loud warning is printed. That is
how a faulty clock state like the one seen on 2026-11-19 (single-digit GB/s) gets caught.
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

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
_PACKAGES = ["numpy", "scipy", "cupy-cuda12x", "cupy-cuda13x", "qiskit", "qiskit-aer",
             "qiskit-aer-gpu", "cuquantum-python-cu12", "cuquantum-python-cu13", "qsim-gpu"]

# Published peak memory bandwidth in GB/s, matched against the GPU name (substring).
# A healthy copy kernel usually reaches 70-90% of the peak, so 50% leaves a wide margin.
PUBLISHED_BANDWIDTH: dict[str, tuple[float, str]] = {
    "Tesla T4": (320.0, "NVIDIA T4 Tensor Core product brief (2020): 'up to 320 GB/s'"),
}
SUSPECT_FRACTION = 0.5


def gpu_slug(name: str | None = None) -> str:
    """Folder name for a GPU, e.g. 'Tesla T4' -> 'tesla-t4'. 'cpu-only' without a GPU."""
    if name is None:
        info = _gpu_info()  # CuPy's name, else nvidia-smi's (environments without CuPy)
        name = str(info.get("name") or info.get("nvidia_smi_name") or "cpu-only")
    return "-".join("".join(c if c.isalnum() else " " for c in name.lower()).split())


def results_dir() -> Path:
    """bench/results/<gpu-slug>/ for the GPU this process runs on."""
    return RESULTS_ROOT / gpu_slug()


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


def bandwidth_check(gpu_name: str | None) -> dict[str, Any]:
    """Time a 64 MiB complex64 copy and compare it with the GPU's published bandwidth."""
    try:
        import cupy as cp
    except Exception as exc:
        return {"status": "no-gpu", "reason": f"{type(exc).__name__}: {exc}"}
    src = cp.ones(1 << 23, dtype=cp.complex64)  # 64 MiB, far larger than any L2 cache
    dst = cp.empty_like(src)
    nbytes = src.nbytes

    def copy() -> None:
        cp.copyto(dst, src)

    t = time_gpu(copy, repeats=7)
    measured = 2 * nbytes / t["median_ms"] / 1e6  # read + write, GB/s
    check: dict[str, Any] = {"copy_64MiB_GBps": measured, "suspect": False}
    for key, (published, source) in PUBLISHED_BANDWIDTH.items():
        if gpu_name and key.lower() in gpu_name.lower():
            check.update(published_GBps=published, published_source=source,
                         fraction_of_published=measured / published,
                         suspect=measured < SUSPECT_FRACTION * published)
            break
    else:
        check["published_GBps"] = None  # no published figure on file: no gate
    cp.get_default_memory_pool().free_all_blocks()
    return check


_BANNER = "!" * 78


def environment() -> dict[str, Any]:
    """Everything needed to reproduce or interpret a benchmark number."""
    versions: dict[str, str | None] = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    gpu = _gpu_info()
    return {
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "git_commit": _git_commit(),
        "gpu": gpu,
        "bandwidth_check": bandwidth_check(gpu.get("name")),
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
    """Write payload + environment to bench/results/<gpu>/<name>.json and return the path.

    The top-level "suspect" flag is True when the bandwidth check failed or the driver reported
    a faulty clock state; such files must not be quoted.
    """
    env = environment()
    check = env["bandwidth_check"]
    suspect = bool(check.get("suspect")) or "warning" in env["gpu"]
    if suspect:
        print(f"{_BANNER}\nWARNING: this GPU looks unhealthy. Copy bandwidth "
              f"{check.get('copy_64MiB_GBps', float('nan')):.1f} GB/s vs published "
              f"{check.get('published_GBps')} GB/s. Results are marked suspect.\n{_BANNER}",
              file=sys.stderr)
    out_dir = results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps({"benchmark": name, "suspect": suspect, "environment": env,
                                **payload}, indent=2) + "\n")
    return path


if __name__ == "__main__":
    env = environment()
    print(json.dumps(env, indent=2))
    out = save_result("env", {})
    print(f"saved {out}")
