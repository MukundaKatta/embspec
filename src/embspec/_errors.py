"""Typed errors for embspec."""

from __future__ import annotations


class EmbspecError(Exception):
    """Base class for all embspec errors."""


class EmbeddingVersionMismatch(EmbspecError):
    """The query encoder's embedding spec does not match the index manifest.

    Raised by :func:`embspec.assert_compatible` and the
    :func:`embspec.embed_assert` decorator. Catching this exception is the
    fast-fail signal that prevents the silent-accuracy-collapse failure
    mode described in the decompressed.io RAG observability post-mortem
    (2026-03-09): the system stays 200 OK with normal latency while
    retrieval quality silently tanks.
    """

    def __init__(
        self,
        *,
        index_name: str,
        manifest_field: str,
        manifest_value: object,
        query_value: object,
        manifest_path: str | None = None,
    ) -> None:
        suffix = f" (manifest at {manifest_path})" if manifest_path else ""
        super().__init__(
            f"Index {index_name!r} manifest declares {manifest_field}="
            f"{manifest_value!r} but query encoder uses {query_value!r}{suffix}. "
            f"Re-encode the corpus or roll the query encoder back."
        )
        self.index_name = index_name
        self.manifest_field = manifest_field
        self.manifest_value = manifest_value
        self.query_value = query_value
        self.manifest_path = manifest_path


class ManifestFormatError(EmbspecError):
    """The manifest file is missing required fields or has an unknown format version."""


class AdapterShapeError(EmbspecError):
    """Embedding tensors have incompatible shapes for fitting or transforming."""
