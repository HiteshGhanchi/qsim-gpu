"""Benchmark tooling: per-GPU folders, the bandwidth gate and the README report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench import report, run_env


def test_gpu_slug() -> None:
    assert run_env.gpu_slug("Tesla T4") == "tesla-t4"
    assert run_env.gpu_slug("NVIDIA GeForce RTX 3050 Laptop GPU") == (
        "nvidia-geforce-rtx-3050-laptop-gpu")


def test_report_builds_from_stored_laptop_results() -> None:
    folder = report.RESULTS / "nvidia-geforce-rtx-3050-laptop-gpu"
    text = report.build(report.Results(folder, allow_suspect=False))
    assert "Copy bandwidth (B0)" in text and "| Circuit | Qubits |" in text


def test_report_refuses_suspect_files(tmp_path: Path) -> None:
    (tmp_path / "b0_bandwidth.json").write_text(json.dumps(
        {"suspect": True, "environment": {"gpu": {}}, "achievable_GBps": 8.0}))
    with pytest.raises(SystemExit):
        report.Results(tmp_path, allow_suspect=False).get("b0_bandwidth")
    assert report.Results(tmp_path, allow_suspect=True).get("b0_bandwidth") is not None


def test_old_files_with_driver_warning_count_as_suspect(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text(json.dumps({"environment": {"gpu": {"warning": "bad"}}}))
    with pytest.raises(SystemExit):
        report.Results(tmp_path, allow_suspect=False).get("x")


@pytest.mark.gpu
def test_bandwidth_gate_flags_a_gpu_far_below_its_published_figure(
        monkeypatch: pytest.MonkeyPatch) -> None:
    name = run_env._gpu_info()["name"]
    monkeypatch.setitem(run_env.PUBLISHED_BANDWIDTH, name, (1e6, "test value"))
    check = run_env.bandwidth_check(name)
    assert check["suspect"] is True and check["published_GBps"] == 1e6
    monkeypatch.setitem(run_env.PUBLISHED_BANDWIDTH, name, (1.0, "test value"))
    assert run_env.bandwidth_check(name)["suspect"] is False
