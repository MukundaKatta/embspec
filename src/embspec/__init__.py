"""embspec — embedding pipeline ops + drift detection for production RAG.

The single primitive this library asserts:

    "The query encoder must use the same embedding model+version as the
    index it is searching."

The decompressed.io RAG observability post-mortem (2026-03-09) describes
exactly the failure this library prevents: the query embedding model gets
upgraded, the index still holds vectors from the old model, every health
check stays 200 OK while retrieval accuracy silently collapses. Reporter
spent $15k on emergency re-encoding and 2-5 days of engineer time before
diagnosis.

Quick start::

    from embspec import IndexManifest, EmbeddingSpec, embed_assert

    # Recorded once when the index was built
    manifest = IndexManifest.load("s3://my-rag/index-prod/manifest.json")

    @embed_assert(manifest, model_id="amazon.titan-embed-text-v2:0", dimension=1024)
    def search(query: str):
        qv = embed_query(query)
        return opensearch.knn_search(qv, ...)

If the query encoder ever drifts off the manifest, ``search`` raises
:class:`EmbeddingVersionMismatch` instead of returning silently-degraded
results.
"""

from __future__ import annotations

from ._adapter import DriftAdapter
from ._assert import embed_assert
from ._errors import (
    AdapterShapeError,
    EmbeddingVersionMismatch,
    EmbspecError,
    ManifestFormatError,
)
from ._manifest import (
    EmbeddingSpec,
    IndexManifest,
    assert_compatible,
)
from ._stability import StabilityReport, neighbor_stability

__version__ = "0.1.0"

__all__ = [
    "AdapterShapeError",
    "DriftAdapter",
    "EmbeddingSpec",
    "EmbeddingVersionMismatch",
    "EmbspecError",
    "IndexManifest",
    "ManifestFormatError",
    "StabilityReport",
    "__version__",
    "assert_compatible",
    "embed_assert",
    "neighbor_stability",
]
