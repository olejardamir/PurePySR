from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from python_backend.options import BackendOptions


@runtime_checkable
class BackendAPI(Protocol):
    def equation_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        options: BackendOptions,
        extra_options: dict[str, Any] | None = None,
        seed: int = 0,
        saved_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


def get_backend(name: str) -> BackendAPI:
    if name == "python":
        from python_backend.backend import PythonSRBackend

        return PythonSRBackend()

    if name == "julia":
        raise NotImplementedError(
            "Julia backend is not yet available through this API. "
            "See BACKEND_SWITCHING.md for migration strategies."
        )

    raise ValueError(f"Unknown backend {name!r}; expected 'python' or 'julia'")
