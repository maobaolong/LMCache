# SPDX-License-Identifier: Apache-2.0
"""Run the exact installed-wheel staged-write kernel without vLLM or LMCache."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
import triton
from triton.backends.cpu.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
source_path = Path(sys.argv[1])
source = source_path.read_text()
tree = ast.parse(source)
functions = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        "_apply_write_kernel", "_load_ptr"
    }:
        functions.append("\n".join(source.splitlines()[node.decorator_list[0].lineno - 1:node.end_lineno]))
kernel_path = source_path.parent / "isolated_staged_write.py"
kernel_path.write_text("import triton\nimport triton.language as tl\n\n" + "\n\n".join(functions) + "\n")
spec = importlib.util.spec_from_file_location("isolated_staged_write", kernel_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

torch.set_num_threads(1)
output = torch.full((4, 2048), -1, dtype=torch.int32)
indices = torch.tensor([1], dtype=torch.int32)
starts = torch.tensor([7], dtype=torch.int32)
contents = torch.tensor([42], dtype=torch.int32)
cu_lens = torch.tensor([1], dtype=torch.int32)
args = (output, output.stride(0), indices, starts, contents, cu_lens, None)
kernel = module._apply_write_kernel.warmup(*args, BLOCK_SIZE=1024, MULTI_GROUP=False, grid=(1, 1))
dest = Path(os.environ.get("REPRO_DUMP_DIR", "/work/kernel-dump"))
dest.mkdir(parents=True, exist_ok=True)
for key, value in kernel.asm.items():
    path = dest / f"kernel.{key}"
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(str(value))
print(json.dumps({"torch": torch.__version__, "triton": triton.__version__, "target": str(triton.runtime.driver.active.get_current_target()), "metadata": str(kernel.metadata), "asm": list(kernel.asm)}, indent=2), flush=True)
print("Launching exact _apply_write_kernel with one int32 staged write", flush=True)
module._apply_write_kernel[(1, 1)](*args, BLOCK_SIZE=1024, MULTI_GROUP=False)
expected = torch.full_like(output, -1)
expected[1, 7] = 42
torch.testing.assert_close(output, expected)
print("PASS: staged write produced the expected tensor", flush=True)
