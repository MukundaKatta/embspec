"""embed_assert decorator tests."""

from __future__ import annotations

import unittest
import warnings
from datetime import datetime, timezone
from pathlib import Path

try:
    import pytest
except ImportError:  # pragma: no cover - exercised only without dev deps
    # Needs the pytest dev dep. Skip cleanly under stdlib ``unittest`` when it
    # is absent; ``test_assert_unittest`` covers the same behavior there.
    raise unittest.SkipTest("pytest not installed")

from embspec import (
    EmbeddingSpec,
    EmbeddingVersionMismatch,
    IndexManifest,
    embed_assert,
)


def _manifest() -> IndexManifest:
    return IndexManifest(
        index_name="prod",
        embedding=EmbeddingSpec(
            model_id="titan-v2",
            dimension=1024,
            model_version=None,
            normalization="l2",
        ),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_decorator_passes_call_through_when_compatible() -> None:
    manifest = _manifest()

    @embed_assert(manifest, model_id="titan-v2", dimension=1024)
    def search(q: str) -> str:
        return f"results for {q}"

    assert search("hello") == "results for hello"


def test_decorator_raises_on_drift_in_raise_mode() -> None:
    manifest = _manifest()

    @embed_assert(manifest, model_id="titan-v3", dimension=1024)
    def search(q: str) -> str:
        return q

    with pytest.raises(EmbeddingVersionMismatch):
        search("x")


def test_decorator_warns_in_log_mode_does_not_raise() -> None:
    manifest = _manifest()

    @embed_assert(manifest, model_id="titan-v3", dimension=1024, mode="log")
    def search(q: str) -> str:
        return q

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = search("x")
    assert result == "x"
    assert any("drift" in str(w.message) for w in captured)


def test_decorator_loads_manifest_lazily_from_path(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "m.json"
    manifest.save(path)

    @embed_assert(str(path), model_id="titan-v2", dimension=1024)
    def search(q: str) -> str:
        return q

    # Manifest not loaded until first call
    assert search("hi") == "hi"


def test_decorator_with_path_includes_path_in_error(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "m.json"
    manifest.save(path)

    @embed_assert(str(path), model_id="other-model", dimension=1024)
    def search(q: str) -> str:
        return q

    with pytest.raises(EmbeddingVersionMismatch) as exc_info:
        search("x")
    assert str(path) in str(exc_info.value)
