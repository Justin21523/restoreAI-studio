from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np

ProgressCallback = Callable[[str, float, str], None]


class ProcessingCancelled(RuntimeError):
    pass


class CancelToken:
    def __init__(self, checker: Callable[[], bool] | None = None) -> None:
        self._event = Event()
        self._checker = checker

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set() or bool(self._checker and self._checker())

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ProcessingCancelled("Processing was cancelled")


@dataclass
class ProcessingResult:
    image: np.ndarray | None = None
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    model_id: str

    @abstractmethod
    def load(self) -> dict[str, Any]: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def health(self) -> dict[str, Any]: ...
