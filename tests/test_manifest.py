"""IndexManifest tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from embspec import (
    EmbeddingSpec,
    EmbeddingVersionMismatch,
    IndexManifest,
    ManifestFormatError,
    assert_compatible,
)


def _make_manifest(**overrides) -> IndexManifest:
    spec = EmbeddingSpec(
        model_id="amazon.titan-embed-text-v2:0",
        dimension=1024,
        model_version=None,
        normalization="l2",
    )
    base = {
        "index_name": "prod-v3",
        "embedding": spec,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "doc_count": 1_000_000,
    }
    base.update(overrides)
    return IndexManifest(**base)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    manifest = _make_manifest()
    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = IndexManifest.load(path)
    assert loaded.index_name == manifest.index_name
    assert loaded.embedding == manifest.embedding
    assert loaded.doc_count == manifest.doc_count


def test_assert_compatible_passes_for_exact_match() -> None:
    manifest = _make_manifest()
    assert_compatible(manifest, manifest.embedding)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("model_id", "openai:text-embedding-3-small"),
        ("dimension", 1536),
        ("model_version", "v2"),
        ("normalization", "none"),
    ],
)
def test_assert_compatible_raises_on_each_field_drift(
    field: str, bad_value: object
) -> None:
    manifest = _make_manifest()
    bad_spec_kwargs = {
        "model_id": manifest.embedding.model_id,
        "dimension": manifest.embedding.dimension,
        "model_version": manifest.embedding.model_version,
        "normalization": manifest.embedding.normalization,
    }
    bad_spec_kwargs[field] = bad_value
    bad = EmbeddingSpec(**bad_spec_kwargs)  # type: ignore[arg-type]
    with pytest.raises(EmbeddingVersionMismatch) as exc_info:
        manifest.assert_compatible(bad)
    assert exc_info.value.manifest_field == f"embedding.{field}"
    assert exc_info.value.index_name == manifest.index_name


def test_load_rejects_unknown_format_version(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "embspec_format_version": 999,
                "index_name": "x",
                "embedding": {"model_id": "x", "dimension": 1},
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    with pytest.raises(ManifestFormatError) as exc_info:
        IndexManifest.load(path)
    assert "999" in str(exc_info.value)


def test_load_rejects_missing_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"embspec_format_version": 1}))
    with pytest.raises(ManifestFormatError):
        IndexManifest.load(path)


def test_exception_message_carries_path_and_values() -> None:
    manifest = _make_manifest()
    bad = EmbeddingSpec(model_id="other", dimension=1024)
    with pytest.raises(EmbeddingVersionMismatch) as exc_info:
        manifest.assert_compatible(bad, manifest_path="s3://bucket/m.json")
    msg = str(exc_info.value)
    assert "s3://bucket/m.json" in msg
    assert "other" in msg
    assert "amazon.titan-embed-text-v2:0" in msg
