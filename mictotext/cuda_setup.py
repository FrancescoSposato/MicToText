"""Make CUDA runtime DLLs (cuBLAS, cuDNN) discoverable on Windows.

CTranslate2 loads cuBLAS/cuDNN dynamically. When they are installed through pip
(nvidia-cublas-cu12, nvidia-cudnn-cu12) the DLLs live in site-packages/nvidia/*/bin,
which is not on the DLL search path. This module must run BEFORE importing
faster_whisper / ctranslate2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mictotext.config import PROJECT_ROOT

_NVIDIA_PACKAGES = ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime")
_registered: list[str] | None = None


def _candidate_dirs() -> list[Path]:
    roots = {Path(p) for p in sys.path if p and p.lower().endswith("site-packages")}
    candidates: list[Path] = []
    for root in roots:
        for package in _NVIDIA_PACKAGES:
            for sub in ("bin", "lib"):
                directory = root / "nvidia" / package / sub
                if directory.is_dir():
                    candidates.append(directory)
    manual_dir = PROJECT_ROOT / "cuda_libs"
    if manual_dir.is_dir():
        candidates.append(manual_dir)
    return candidates


def register_cuda_dll_dirs() -> list[str]:
    """Register CUDA DLL folders once per process. No-op outside Windows."""
    global _registered
    if _registered is not None:
        return _registered
    _registered = []
    if sys.platform != "win32":
        return _registered

    for directory in dict.fromkeys(_candidate_dirs()):
        try:
            os.add_dll_directory(str(directory))
        except OSError:
            continue
        _registered.append(str(directory))

    if _registered:
        # Some native loaders ignore add_dll_directory and only look at PATH.
        os.environ["PATH"] = os.pathsep.join(_registered + [os.environ.get("PATH", "")])
    return _registered
