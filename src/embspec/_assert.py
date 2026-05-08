"""Decorator for asserting embedding-spec compatibility on every retrieval call."""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

from ._manifest import EmbeddingSpec, IndexManifest


F = TypeVar("F", bound=Callable[..., Any])


def embed_assert(
    manifest: IndexManifest | str | Path,
    *,
    model_id: str,
    dimension: int,
    model_version: str | None = None,
    normalization: Literal["l2", "none"] = "l2",
    mode: Literal["raise", "log"] = "raise",
) -> Callable[[F], F]:
    """Assert the function uses an embedding spec compatible with the index manifest.

    ``manifest`` may be an :class:`IndexManifest` or a path the manifest is
    loaded from (the path is resolved on first call so the decorated symbol
    stays import-cheap).

    ``mode="raise"`` (default) raises :class:`EmbeddingVersionMismatch` on
    drift. ``mode="log"`` is the safer rollout mode: it never raises but
    logs via :mod:`warnings` so production stays up while you wire alerts.
    """
    spec = EmbeddingSpec(
        model_id=model_id,
        dimension=dimension,
        model_version=model_version,
        normalization=normalization,
    )

    state: dict[str, Any] = {"resolved": None}

    def _resolve() -> tuple[IndexManifest, str | None]:
        if state["resolved"] is None:
            if isinstance(manifest, IndexManifest):
                state["resolved"] = (manifest, None)
            else:
                path = str(manifest)
                state["resolved"] = (IndexManifest.load(path), path)
        return state["resolved"]

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            m, path = _resolve()
            try:
                m.assert_compatible(spec, manifest_path=path)
            except Exception:
                if mode == "log":
                    import warnings

                    warnings.warn(
                        f"embspec drift on call to {fn.__qualname__}",
                        stacklevel=2,
                    )
                else:
                    raise
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
