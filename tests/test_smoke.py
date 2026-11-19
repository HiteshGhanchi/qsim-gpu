"""Phase 0 smoke tests: the package imports and the GPU toolchain can build and run a kernel."""

import numpy as np
import pytest

import qsim
from qsim.kernels import build_source, load_source


def test_package_imports() -> None:
    assert qsim.__version__


def test_kernel_sources_are_packaged() -> None:
    assert "copy_state" in load_source("gates.cu")
    assert "insert_zero_bits" in load_source("common.cuh")


def test_build_source_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError):
        build_source("gates.cu", "float16")


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_copy_kernel_runs_on_gpu(dtype: type) -> None:
    import cupy as cp

    module = cp.RawModule(code=build_source("gates.cu", np.dtype(dtype).name))
    kernel = module.get_function("copy_state")
    n = 1 << 12
    rng = np.random.default_rng(0)
    host = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dtype)
    src = cp.asarray(host)
    dst = cp.zeros_like(src)
    kernel((64,), (256,), (src, dst, np.int64(n)))
    np.testing.assert_array_equal(dst.get(), host)
