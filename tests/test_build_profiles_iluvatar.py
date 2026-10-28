# SPDX-License-Identifier: Apache-2.0
"""CoreX selection and shared CUDA extension build contracts (no GPU required)."""

# Standard
from unittest.mock import Mock
import os
import sys

# Third Party
from setuptools import Extension
from torch.utils import cpp_extension
import pytest
import torch

# First Party
from setup_extensions import BuildPolicy
from setup_extensions.build_profiles.cuda import CudaProfile
from setup_extensions.build_profiles.iluvatar import IluvatarProfile
from setup_extensions.common_cpp import CommonExtSpec


@pytest.fixture(autouse=True)
def clean_build_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate explicit selection from the enclosing CI lane's environment."""
    for name in list(os.environ):
        if name.startswith("BUILD_WITH_") or name in (
            "NO_NATIVE_EXT",
            "NO_CUDA_EXT",
            "NO_GPU_EXT",
            "MAX_JOBS",
            "NVCC_THREADS",
        ):
            monkeypatch.delenv(name)


@pytest.fixture
def extension_factory(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Capture setuptools inputs without compiling or probing a GPU."""
    factory = Mock(return_value=Extension("lmcache.cuda_ops", sources=[]))
    monkeypatch.setattr(cpp_extension, "CUDAExtension", factory)
    return factory


def test_explicit_selection_on_a_headless_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BUILD_WITH_ILUVATAR", "1")
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setitem(sys.modules, "lmcache", None)
    monkeypatch.setitem(sys.modules, "torch", None)

    assert isinstance(BuildPolicy().resolve_profile(), IluvatarProfile)


def test_compiler_detection_needs_no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(
        "shutil.which", lambda name: "/corex/bin/nvcc" if name == "nvcc" else None
    )

    assert IluvatarProfile().detect()


def test_generic_cuda_auto_selection_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "shutil.which", lambda name: "/toolchain/bin/nvcc" if name == "nvcc" else None
    )

    assert type(BuildPolicy().resolve_profile()) is CudaProfile


def test_conflicting_explicit_selections_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUILD_WITH_CUDA", "1")
    monkeypatch.setenv("BUILD_WITH_ILUVATAR", "1")

    with pytest.raises(RuntimeError, match="Multiple profiles explicitly requested"):
        BuildPolicy().resolve_profile()


@pytest.mark.parametrize("abi", [False, True])
def test_corex_shares_sources_and_selects_portable_io(
    monkeypatch: pytest.MonkeyPatch, extension_factory: Mock, abi: bool
) -> None:
    monkeypatch.setenv("MAX_JOBS", "1")
    monkeypatch.setenv("NVCC_THREADS", "4")
    CudaProfile().build()
    cuda_inputs = extension_factory.call_args.kwargs
    monkeypatch.setattr(torch, "compiled_with_cxx11_abi", lambda: abi)

    extensions, commands = IluvatarProfile().build()
    corex_inputs = extension_factory.call_args.kwargs

    assert extensions == [extension_factory.return_value]
    assert commands == {"build_ext": cpp_extension.BuildExtension}
    assert extension_factory.call_args.args == ("lmcache.cuda_ops",)
    assert corex_inputs["sources"] == cuda_inputs["sources"]
    assert "csrc/cuda/phase_timing_recorder.cu" in corex_inputs["sources"]
    assert corex_inputs["include_dirs"] == cuda_inputs["include_dirs"]
    assert corex_inputs["define_macros"] == [("LMCACHE_DISABLE_STREAMING_IO", "1")]
    assert corex_inputs["extra_compile_args"] == {
        "cxx": [f"-D_GLIBCXX_USE_CXX11_ABI={int(abi)}"],
        "nvcc": [f"-D_GLIBCXX_USE_CXX11_ABI={int(abi)}"],
    }
    assert os.environ["MAX_JOBS"] == "1"
    common_spec = CommonExtSpec("test", sources=[], include_dirs=[])
    assert IluvatarProfile().extra_cxx_flags_for(common_spec) == [
        f"-D_GLIBCXX_USE_CXX11_ABI={int(abi)}"
    ]


def test_nvidia_build_keeps_streaming_io_and_compiler_threads(
    monkeypatch: pytest.MonkeyPatch, extension_factory: Mock
) -> None:
    monkeypatch.setenv("MAX_JOBS", "1")
    monkeypatch.setenv("NVCC_THREADS", "4")

    CudaProfile().build()
    inputs = extension_factory.call_args.kwargs

    assert inputs["define_macros"] == []
    assert "--threads=4" in inputs["extra_compile_args"]["nvcc"]
    assert inputs["extra_compile_args"]["cxx"] == CudaProfile().default_cxx_flags()


def test_corex_does_not_request_nvidia_runtime_packages() -> None:
    assert IluvatarProfile().requirements_file() is None
    assert IluvatarProfile().extras_requirements() == {}
