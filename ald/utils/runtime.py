import importlib.metadata as metadata
import platform
import random
import subprocess
from contextlib import nullcontext

import numpy as np
import torch


def versions():
    packages = (
        "torch",
        "numpy",
        "PyYAML",
        "transformers",
        "huggingface-hub",
        "tokenizers",
        "safetensors",
        "peft",
        "accelerate",
        "pytest",
        "gradio",
        "gradio-client",
        "umap-learn",
        "numba",
        "scikit-learn",
        "scipy",
        "pandas",
    )
    installed = {}
    for package in packages:
        try:
            installed[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed[package] = None
    installed["torch"] = str(torch.__version__)
    return installed


def environment():
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": versions(),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        report["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv",
            ],
            text=True,
            timeout=8,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        report["nvidia_smi"] = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report["gpu"] = {
            "name": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "compiled_arches": torch.cuda.get_arch_list(),
            "free_mib": round(free / 2**20),
            "total_mib": round(total / 2**20),
            "bf16": torch.cuda.is_bf16_supported(including_emulation=False),
        }
        try:
            dtype = torch.bfloat16 if report["gpu"]["bf16"] else torch.float16
            vector = torch.ones((16, 16), device="cuda", dtype=dtype)
            report["gpu"]["kernel_ok"] = (vector @ vector)[0, 0].item() == 16
        except RuntimeError as exc:
            report["gpu"]["kernel_ok"] = False
            report["gpu"]["kernel_error"] = str(exc)
    return report


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def generator(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed))


class Runtime:
    def __init__(self, config, device_override=None):
        requested = device_override or config["device"]
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Use cpu or cuda.")
        precision = config["precision"]
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but PyTorch cannot access it.")
            with torch.cuda.device(self.device):
                bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
            if precision in {"auto", "bf16"}:
                precision = "bf16" if bf16 else "fp16"
        else:
            precision = "fp32"
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
            precision
        ]
        if config["backend"] == "mock" and self.device.type == "cpu":
            torch.set_num_threads(min(2, torch.get_num_threads()))

    def autocast(self):
        if self.dtype == torch.float32:
            return nullcontext()
        return torch.autocast(self.device.type, dtype=self.dtype)

    def reset_peak(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def memory(self):
        if self.device.type == "cpu":
            return {}
        return {
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(self.device) / 2**20, 2),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(self.device) / 2**20, 2),
        }
