"""Stdlib ``unittest`` coverage for the ``embed_assert`` decorator.

Mirrors ``test_assert.py`` using only the standard library so the suite runs
under ``python3 -m unittest discover -s tests`` without pytest installed.
"""

from __future__ import annotations

import tempfile
import unittest
import warnings
from datetime import datetime, timezone
from pathlib import Path

try:  # ensure src/ is importable under the stdlib runner (no-op once installed)
    from . import _path  # noqa: F401
except ImportError:
    import _path  # noqa: F401

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


class EmbedAssertTests(unittest.TestCase):
    def test_passes_call_through_when_compatible(self) -> None:
        manifest = _manifest()

        @embed_assert(manifest, model_id="titan-v2", dimension=1024)
        def search(q: str) -> str:
            return f"results for {q}"

        self.assertEqual(search("hello"), "results for hello")

    def test_raises_on_drift_in_raise_mode(self) -> None:
        manifest = _manifest()

        @embed_assert(manifest, model_id="titan-v3", dimension=1024)
        def search(q: str) -> str:
            return q

        with self.assertRaises(EmbeddingVersionMismatch):
            search("x")

    def test_warns_in_log_mode_does_not_raise(self) -> None:
        manifest = _manifest()

        @embed_assert(manifest, model_id="titan-v3", dimension=1024, mode="log")
        def search(q: str) -> str:
            return q

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = search("x")
        self.assertEqual(result, "x")
        self.assertTrue(any("drift" in str(w.message) for w in captured))

    def test_preserves_wrapped_function_metadata(self) -> None:
        manifest = _manifest()

        @embed_assert(manifest, model_id="titan-v2", dimension=1024)
        def search(q: str) -> str:
            """Docstring marker."""
            return q

        self.assertEqual(search.__name__, "search")
        self.assertEqual(search.__doc__, "Docstring marker.")

    def test_loads_manifest_lazily_from_path(self) -> None:
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            manifest.save(path)

            @embed_assert(str(path), model_id="titan-v2", dimension=1024)
            def search(q: str) -> str:
                return q

            self.assertEqual(search("hi"), "hi")

    def test_path_included_in_error(self) -> None:
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            manifest.save(path)

            @embed_assert(str(path), model_id="other-model", dimension=1024)
            def search(q: str) -> str:
                return q

            with self.assertRaises(EmbeddingVersionMismatch) as cm:
                search("x")
            self.assertIn(str(path), str(cm.exception))


if __name__ == "__main__":
    unittest.main()
