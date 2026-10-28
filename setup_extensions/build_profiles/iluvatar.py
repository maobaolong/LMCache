# SPDX-License-Identifier: Apache-2.0
"""Explicit CoreX build using the shared CUDA source and extension definition.

Select with ``BUILD_WITH_ILUVATAR=1`` in a vendor-provided CoreX/PyTorch
environment, including on headless hosts. Inherited detection only checks
for the nvcc-compatible compiler; it does not identify the vendor. Generic
CUDA auto-detection remains unchanged, so explicit selection is required.
"""

# First Party
from setup_extensions.build_profiles.cuda import CudaProfile


class IluvatarProfile(CudaProfile):
    """CoreX compiler capabilities; the runtime continues to use torch.cuda."""

    name = "iluvatar"
    env_var = "BUILD_WITH_ILUVATAR"

    def cuda_defines(self) -> list[tuple[str, str]]:
        """Select ordinary vector I/O because CoreX cannot compile streaming PTX."""
        return [("LMCACHE_DISABLE_STREAMING_IO", "1")]

    def nvcc_flags(self) -> list[str]:
        """Return ABI flags without NVIDIA's compiler-internal --threads option.

        Set ``MAX_JOBS`` explicitly to bound translation-unit parallelism.
        ``NVCC_THREADS`` applies only to the NVIDIA compiler.
        """
        return self.default_cxx_flags()

    def default_cxx_flags(self) -> list[str]:
        """Match the ABI of the installed CoreX PyTorch for all extensions."""
        # Third Party
        import torch

        return [f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}"]

    def requirements_file(self) -> str | None:
        """Use the preinstalled vendor runtime rather than NVIDIA CUDA wheels."""
        return None

    def extras_requirements(self) -> dict[str, str]:
        """Return no NVIDIA-specific optional extras for the CoreX build."""
        return {}
