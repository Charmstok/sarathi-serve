"""Sarathi: a high-throughput and memory-efficient inference engine for LLMs"""

import os
import shutil
import subprocess
import sys
from packaging.version import Version


def _get_nvcc_candidates() -> list[str]:
    candidates: list[str] = []
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(os.path.join(cuda_home, "bin", "nvcc"))
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(nvcc)
    for prefix in ("/usr/local/cuda", "/usr/local/cuda-12.8", "/usr/local/cuda-12.2"):
        candidates.append(os.path.join(prefix, "bin", "nvcc"))
    # dedupe while preserving order
    return list(dict.fromkeys(candidates))


def _get_nvcc_version() -> Version | None:
    for candidate in _get_nvcc_candidates():
        if not os.path.exists(candidate):
            continue
        try:
            output = subprocess.check_output([candidate, "--version"], text=True)
        except Exception:
            continue
        for token in output.replace(",", " ").split():
            if token.count(".") == 1 and token[0].isdigit():
                try:
                    return Version(token)
                except Exception:
                    continue
    return None


def _get_first_gpu_compute_capability_major() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
        )
    except Exception:
        return None

    first_line = output.strip().splitlines()[0] if output.strip() else ""
    if not first_line:
        return None
    try:
        major_str, _minor_str = first_line.split(".", 1)
        return int(major_str)
    except Exception:
        return None



def _prepend_env_path(var_name: str, path_value: str) -> None:
    current = os.environ.get(var_name, "")
    parts = [p for p in current.split(":") if p]
    if path_value in parts:
        return
    os.environ[var_name] = ":".join([path_value] + parts) if parts else path_value


def _configure_python_tooling_paths() -> None:
    python_bin_dir = os.path.dirname(sys.executable)
    if os.path.isdir(python_bin_dir):
        _prepend_env_path("PATH", python_bin_dir)


def _configure_cuda_library_paths() -> None:
    libcuda_candidates = [
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64/stubs",
        "/usr/local/cuda-12.2/targets/x86_64-linux/lib/stubs",
    ]
    for candidate in libcuda_candidates:
        if os.path.exists(os.path.join(candidate, "libcuda.so")) or os.path.exists(
            os.path.join(candidate, "libcuda.so.1")
        ):
            _prepend_env_path("LIBRARY_PATH", candidate)
            _prepend_env_path("LD_LIBRARY_PATH", candidate)



def _configure_flashinfer_runtime() -> None:
    # FlashInfer 0.4.1 tries to JIT Blackwell kernels with sm_120a. This host only
    # has CUDA 12.2 nvcc, which cannot compile that target. We fall back to Hopper
    # PTX so the driver can JIT on Blackwell at runtime.
    major = _get_first_gpu_compute_capability_major()
    if major is None:
        return

    nvcc_candidates = _get_nvcc_candidates()
    for candidate in nvcc_candidates:
        if os.path.exists(candidate):
            os.environ.setdefault("FLASHINFER_NVCC", candidate)
            os.environ.setdefault("CUDA_HOME", os.path.dirname(os.path.dirname(candidate)))
            break

    nvcc_version = _get_nvcc_version()
    if major >= 10 and nvcc_version is not None and nvcc_version < Version("12.8"):
        os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "9.0")
        extra_cuda_flags = os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
        ptx_flag = "-gencode=arch=compute_90,code=compute_90"
        if ptx_flag not in extra_cuda_flags.split():
            os.environ["FLASHINFER_EXTRA_CUDAFLAGS"] = (extra_cuda_flags + " " + ptx_flag).strip()

    _configure_python_tooling_paths()
    _configure_cuda_library_paths()

    gcc12 = shutil.which("gcc-12")
    gpp12 = shutil.which("g++-12")
    if gcc12 and gpp12:
        os.environ.setdefault("CC", gcc12)
        os.environ.setdefault("CXX", gpp12)
        os.environ.setdefault("CUDAHOSTCXX", gpp12)


_configure_flashinfer_runtime()

from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.engine.llm_engine import LLMEngine

__version__ = "0.1.8"

__all__ = [
    "SamplingParams",
    "RequestOutput",
    "LLMEngine",
]
