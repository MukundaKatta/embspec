"""Stdlib ``unittest`` coverage for IndexManifest / EmbeddingSpec.

These mirror ``test_manifest.py`` but use only the standard library so the
suite is runnable with ``python3 -m unittest discover -s tests`` in an
environment without pytest or numpy installed. They exercise the real
``embspec`` manifest + assertion code, including regression tests for the
typed-error and naive-datetime fixes.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:  # ensure src/ is importable under the stdlib runner (no-op once installed)
    from . import _path  # noqa: F401
except ImportError:
    import _path  # noqa: F401

from embspec import (
    EmbeddingSpec,
    EmbeddingVersionMismatch,
    IndexManifest,
    ManifestFormatError,
    assert_compatible,
)


def _make_manifest(**overrides: object) -> IndexManifest:
    spec = EmbeddingSpec(
        model_id="amazon.titan-embed-text-v2:0",
        dimension=1024,
        model_version=None,
        normalization="l2",
    )
    base: dict[str, object] = {
        "index_name": "prod-v3",
        "embedding": spec,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "doc_count": 1_000_000,
    }
    base.update(overrides)
    return IndexManifest(**base)  # type: ignore[arg-type]


class SaveLoadTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        manifest = _make_manifest()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "manifest.json"
            manifest.save(path)
            loaded = IndexManifest.load(path)
        self.assertEqual(loaded.index_name, manifest.index_name)
        self.assertEqual(loaded.embedding, manifest.embedding)
        self.assertEqual(loaded.doc_count, manifest.doc_count)
        self.assertEqual(loaded.created_at, manifest.created_at)

    def test_to_dict_includes_format_version(self) -> None:
        d = _make_manifest().to_dict()
        self.assertEqual(d["embspec_format_version"], 1)
        self.assertEqual(d["index_name"], "prod-v3")
        self.assertEqual(d["embedding"]["dimension"], 1024)

    def test_extra_defaults_to_empty_dict_and_roundtrips(self) -> None:
        manifest = _make_manifest(extra={"git_sha": "abc123"})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            manifest.save(path)
            loaded = IndexManifest.load(path)
        self.assertEqual(loaded.extra, {"git_sha": "abc123"})


class AssertCompatibleTests(unittest.TestCase):
    def test_passes_for_exact_match(self) -> None:
        manifest = _make_manifest()
        # Should not raise.
        assert_compatible(manifest, manifest.embedding)
        manifest.assert_compatible(manifest.embedding)

    def test_raises_on_each_field_drift(self) -> None:
        manifest = _make_manifest()
        cases = {
            "model_id": "openai:text-embedding-3-small",
            "dimension": 1536,
            "model_version": "v2",
            "normalization": "none",
        }
        for field, bad_value in cases.items():
            with self.subTest(field=field):
                kwargs = {
                    "model_id": manifest.embedding.model_id,
                    "dimension": manifest.embedding.dimension,
                    "model_version": manifest.embedding.model_version,
                    "normalization": manifest.embedding.normalization,
                }
                kwargs[field] = bad_value
                bad = EmbeddingSpec(**kwargs)  # type: ignore[arg-type]
                with self.assertRaises(EmbeddingVersionMismatch) as cm:
                    manifest.assert_compatible(bad)
                self.assertEqual(cm.exception.manifest_field, f"embedding.{field}")
                self.assertEqual(cm.exception.index_name, manifest.index_name)

    def test_exception_message_carries_path_and_values(self) -> None:
        manifest = _make_manifest()
        bad = EmbeddingSpec(model_id="other", dimension=1024)
        with self.assertRaises(EmbeddingVersionMismatch) as cm:
            manifest.assert_compatible(bad, manifest_path="s3://bucket/m.json")
        msg = str(cm.exception)
        self.assertIn("s3://bucket/m.json", msg)
        self.assertIn("other", msg)
        self.assertIn("amazon.titan-embed-text-v2:0", msg)


class FormatErrorTests(unittest.TestCase):
    def _from_dict(self, data: dict[str, object]) -> IndexManifest:
        return IndexManifest.from_dict(data)

    def test_rejects_unknown_format_version(self) -> None:
        with self.assertRaises(ManifestFormatError) as cm:
            self._from_dict(
                {
                    "embspec_format_version": 999,
                    "index_name": "x",
                    "embedding": {"model_id": "x", "dimension": 1},
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            )
        self.assertIn("999", str(cm.exception))

    def test_rejects_missing_top_level_fields(self) -> None:
        with self.assertRaises(ManifestFormatError):
            self._from_dict({"embspec_format_version": 1})

    def test_rejects_embedding_missing_required_subfields(self) -> None:
        # Regression: previously leaked a raw KeyError to the caller.
        for emb in ({"dimension": 8}, {"model_id": "m"}):
            with self.subTest(emb=emb):
                with self.assertRaises(ManifestFormatError):
                    self._from_dict(
                        {
                            "embspec_format_version": 1,
                            "index_name": "x",
                            "embedding": emb,
                            "created_at": "2026-01-01T00:00:00+00:00",
                        }
                    )

    def test_rejects_non_numeric_dimension(self) -> None:
        # Regression: previously leaked a raw ValueError from int("abc").
        with self.assertRaises(ManifestFormatError):
            self._from_dict(
                {
                    "embspec_format_version": 1,
                    "index_name": "x",
                    "embedding": {"model_id": "m", "dimension": "abc"},
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            )

    def test_rejects_non_dict_embedding(self) -> None:
        with self.assertRaises(ManifestFormatError):
            self._from_dict(
                {
                    "embspec_format_version": 1,
                    "index_name": "x",
                    "embedding": "not-an-object",
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            )

    def test_rejects_unparseable_created_at(self) -> None:
        # Regression: previously leaked a raw ValueError from fromisoformat.
        for bad in ("not-a-date", 12345):
            with self.subTest(created_at=bad):
                with self.assertRaises(ManifestFormatError):
                    self._from_dict(
                        {
                            "embspec_format_version": 1,
                            "index_name": "x",
                            "embedding": {"model_id": "m", "dimension": 8},
                            "created_at": bad,
                        }
                    )

    def test_load_surfaces_source_path_in_error(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            path.write_text(json.dumps({"embspec_format_version": 1}))
            with self.assertRaises(ManifestFormatError) as cm:
                IndexManifest.load(path)
            self.assertIn(str(path), str(cm.exception))


class NaiveDatetimeTests(unittest.TestCase):
    def test_naive_created_at_serialized_as_utc_not_local(self) -> None:
        # Regression: a naive datetime used to be interpreted as the host's
        # local zone and silently shifted by its UTC offset on serialization.
        manifest = _make_manifest(created_at=datetime(2026, 1, 1, 0, 0, 0))
        self.assertEqual(
            manifest.to_dict()["created_at"], "2026-01-01T00:00:00+00:00"
        )

    def test_naive_created_at_roundtrips_unchanged(self) -> None:
        manifest = _make_manifest(created_at=datetime(2026, 6, 7, 12, 30, 0))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            manifest.save(path)
            loaded = IndexManifest.load(path)
        self.assertEqual(
            loaded.created_at,
            datetime(2026, 6, 7, 12, 30, 0, tzinfo=timezone.utc),
        )

    def test_aware_created_at_normalized_to_utc(self) -> None:
        plus_five = timezone(timedelta(hours=5))
        manifest = _make_manifest(
            created_at=datetime(2026, 1, 1, 5, 0, 0, tzinfo=plus_five)
        )
        self.assertEqual(
            manifest.to_dict()["created_at"], "2026-01-01T00:00:00+00:00"
        )


if __name__ == "__main__":
    unittest.main()
