# SPDX-License-Identifier: Apache-2.0
"""Exercise the original vLLM public staging API without a model or connector."""
import torch
import vllm.v1.worker.cpu.shm
from vllm.platforms import current_platform
current_platform.import_kernels()
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

torch.set_num_threads(2)
print("Constructing one-dimensional int32 StagedWriteTensor", flush=True)
tensor = StagedWriteTensor(4, torch.int32, torch.device("cpu"))
tensor.stage_write_elem(3, 1)
print("Calling StagedWriteTensor.apply_write", flush=True)
tensor.apply_write()
torch.testing.assert_close(tensor.gpu, torch.tensor([0, 0, 0, 1], dtype=torch.int32))
print("PASS: vLLM StagedWriteTensor one-dimensional write", flush=True)
tensor = StagedWriteTensor((4, 2048), torch.int32, torch.device("cpu"))
tensor.stage_write(3, 0, [1])
tensor.apply_write()
expected = torch.zeros((4, 2048), dtype=torch.int32)
expected[3, 0] = 1
torch.testing.assert_close(tensor.gpu, expected)
print("PASS: vLLM StagedWriteTensor two-dimensional write", flush=True)
