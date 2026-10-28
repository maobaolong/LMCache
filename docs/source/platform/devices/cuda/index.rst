CUDA
====

CoreX source builds
-------------------

Iluvatar CoreX exposes ``torch.cuda``. Its build profile produces
``lmcache.cuda_ops`` using the shared CUDA sources, so runtime device selection
continues to use ``CudaDeviceSpec`` and MP AUTO keeps its CUDA routing.

Use a vendor-provided CoreX SDK and matching PyTorch environment with
``nvcc`` on ``PATH`` and the required headers and libraries configured. Install
the dependencies in ``requirements/common.txt`` in that environment, then run
from the LMCache checkout:

.. code-block:: bash

   BUILD_WITH_ILUVATAR=1 MAX_JOBS=1 pip install -e . --no-build-isolation
   python -c "import lmcache.cuda_ops"

Explicit selection is required, including on headless builders. Automatic
selection still chooses the generic CUDA profile when ``nvcc`` is present;
it does not infer a vendor from the attached GPU. Do not also set
``BUILD_WITH_CUDA=1``. The CoreX profile uses the installed PyTorch C++ ABI,
does not append ``cuda12_core.txt`` / ``cuda13_core.txt`` or NVIDIA-specific
optional extras, and leaves vendor runtime packages to the vendor environment.
Common dependencies still
apply; this profile does not provide a complete CoreX installation recipe.

CoreX builds disable streaming PTX through the existing
``LMCACHE_DISABLE_STREAMING_IO`` capability macro. Kernels call shared vector
load/store primitives whose portable implementation uses ordinary reads and
writes. ``NVCC_THREADS`` is NVIDIA-specific and is not forwarded to CoreX;
``MAX_JOBS`` controls translation-unit parallelism. NVIDIA builds retain their
streaming PTX and automatic parallelism settings.

Validation on accelerator hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standalone smoke test needs only the SDK, a C++17-compatible compiler and a
visible GPU. Run it from the repository root with the compiler targeting the
installed GPU (for NVIDIA H200, add ``-arch=sm_90``):

.. code-block:: bash

   nvcc -std=c++17 -I csrc -DLMCACHE_DISABLE_STREAMING_IO=1 \
     tests/cpp/test_streaming_io.cu -o /tmp/lmcache-portable-io
   /tmp/lmcache-portable-io

On NVIDIA, also compile and run without the ``-D`` option to exercise streaming
PTX. Both binaries must report ``streaming I/O roundtrip passed (259 vectors)``.
This checks the vector primitives, not full KV transfer or inference. CoreX
compiler acceptance, full extension import, layout-specific KV roundtrips,
IPC compatibility, second-request cache hits and transfer bandwidth must be
validated on BI-V150 before claiming production support. Advanced CUDA IPC,
VMM, copy-engine and GDS features are not established by this build profile.
