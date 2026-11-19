"""Shared pytest setup.

GPU tests are marked @pytest.mark.gpu. If CuPy is missing or no GPU can be used,
they are skipped instead of failing, so the CPU suite runs anywhere.
"""

from __future__ import annotations

import pytest


def _gpu_available() -> bool:
    try:
        import cupy as cp

        return bool(cp.cuda.runtime.getDeviceCount() > 0 and cp.zeros(1).size == 1)
    except Exception:
        return False


GPU_AVAILABLE = _gpu_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if GPU_AVAILABLE:
        return
    skip_gpu = pytest.mark.skip(reason="no usable GPU / CuPy")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
