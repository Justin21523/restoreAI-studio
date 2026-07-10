from __future__ import annotations

import ctypes
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def preload_nvrtc_builtins() -> Path | None:
    """Preload pip-installed CUDA builtins before NVRTC compiles JIT kernels."""
    candidates: list[Path] = []
    for entry in map(Path, sys.path):
        candidates.extend(entry.glob("nvidia/cu13/lib/libnvrtc-builtins.so.*"))
    for path in sorted(candidates, reverse=True):
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            return path
        except OSError:
            continue
    return None
