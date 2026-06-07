"""Index manifests track what embedding model + version a vector index was built with.

The manifest is the single source of truth for "what does this index contain";
asserting against it at every search prevents the silent failure mode where
a query encoder upgrade ships before the index is re-encoded and accuracy
collapses while every health check stays green.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ._errors import EmbeddingVersionMismatch, ManifestFormatError


_FORMAT_VERSION: int = 1


@dataclass(frozen=True)
class EmbeddingSpec:
    """The embedding configuration used to produce a set of vectors."""

    model_id: str
    dimension: int
    model_version: str | None = None
    normalization: Literal["l2", "none"] = "l2"


@dataclass(frozen=True)
class IndexManifest:
    """Manifest describing the embedding configuration of a vector index."""

    index_name: str
    embedding: EmbeddingSpec
    created_at: datetime
    doc_count: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # A naive ``created_at`` (no tzinfo) is treated as UTC rather than the
        # machine's local zone. ``datetime.astimezone`` would otherwise assume
        # local time and silently shift the timestamp by the host's UTC offset,
        # so the *same* manifest serialized on machines in different timezones
        # would produce different ``created_at`` values — a non-reproducible,
        # silent corruption of the field this library exists to make trustworthy.
        created_at = self.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return {
            "embspec_format_version": _FORMAT_VERSION,
            "index_name": self.index_name,
            "embedding": asdict(self.embedding),
            "created_at": created_at.astimezone(timezone.utc).isoformat(),
            "doc_count": self.doc_count,
            "extra": self.extra,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> IndexManifest:
        return cls.from_dict(json.loads(Path(path).read_text()), source=str(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str | None = None) -> IndexManifest:
        version = data.get("embspec_format_version")
        if version != _FORMAT_VERSION:
            raise ManifestFormatError(
                f"Unknown embspec_format_version={version!r} in manifest"
                + (f" at {source}" if source else "")
            )
        if "embedding" not in data or "index_name" not in data:
            raise ManifestFormatError(
                f"Manifest missing required fields"
                + (f" at {source}" if source else "")
            )
        at_source = f" at {source}" if source else ""
        emb = data["embedding"]
        if not isinstance(emb, dict):
            raise ManifestFormatError(
                f"Manifest 'embedding' must be an object, got {type(emb).__name__}{at_source}"
            )
        # Corrupt field *values* (missing model_id/dimension, a non-numeric
        # dimension, an unparseable created_at) previously leaked raw KeyError
        # / ValueError to callers. The whole point of from_dict is to surface
        # one typed ManifestFormatError for any malformed manifest, so callers
        # have a single exception to catch.
        try:
            embedding = EmbeddingSpec(
                model_id=emb["model_id"],
                dimension=int(emb["dimension"]),
                model_version=emb.get("model_version"),
                normalization=emb.get("normalization", "l2"),
            )
        except KeyError as exc:
            raise ManifestFormatError(
                f"Manifest 'embedding' missing required field {exc.args[0]!r}{at_source}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ManifestFormatError(
                f"Manifest 'embedding' has an invalid field value: {exc}{at_source}"
            ) from exc
        try:
            created_at = (
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now(timezone.utc)
            )
        except (TypeError, ValueError) as exc:
            raise ManifestFormatError(
                f"Manifest 'created_at' is not a valid ISO-8601 timestamp: {exc}{at_source}"
            ) from exc
        return cls(
            index_name=data["index_name"],
            embedding=embedding,
            created_at=created_at,
            doc_count=data.get("doc_count"),
            extra=data.get("extra") or {},
        )

    def assert_compatible(
        self,
        spec: EmbeddingSpec,
        *,
        manifest_path: str | None = None,
    ) -> None:
        """Raise :class:`EmbeddingVersionMismatch` if ``spec`` is not the same as this manifest's embedding.

        Compatibility is exact-match across every field of :class:`EmbeddingSpec`
        — query and index must use the same model, dimension, version, and
        normalization. Any drift on any field is a failure mode.
        """
        if spec.model_id != self.embedding.model_id:
            raise EmbeddingVersionMismatch(
                index_name=self.index_name,
                manifest_field="embedding.model_id",
                manifest_value=self.embedding.model_id,
                query_value=spec.model_id,
                manifest_path=manifest_path,
            )
        if spec.dimension != self.embedding.dimension:
            raise EmbeddingVersionMismatch(
                index_name=self.index_name,
                manifest_field="embedding.dimension",
                manifest_value=self.embedding.dimension,
                query_value=spec.dimension,
                manifest_path=manifest_path,
            )
        if spec.model_version != self.embedding.model_version:
            raise EmbeddingVersionMismatch(
                index_name=self.index_name,
                manifest_field="embedding.model_version",
                manifest_value=self.embedding.model_version,
                query_value=spec.model_version,
                manifest_path=manifest_path,
            )
        if spec.normalization != self.embedding.normalization:
            raise EmbeddingVersionMismatch(
                index_name=self.index_name,
                manifest_field="embedding.normalization",
                manifest_value=self.embedding.normalization,
                query_value=spec.normalization,
                manifest_path=manifest_path,
            )


def assert_compatible(
    manifest: IndexManifest,
    spec: EmbeddingSpec,
    *,
    manifest_path: str | None = None,
) -> None:
    """Functional alias for :meth:`IndexManifest.assert_compatible`."""
    manifest.assert_compatible(spec, manifest_path=manifest_path)
