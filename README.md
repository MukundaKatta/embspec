# embspec

[![PyPI version](https://img.shields.io/pypi/v/embspec.svg)](https://pypi.org/project/embspec/)
[![Python versions](https://img.shields.io/pypi/pyversions/embspec.svg)](https://pypi.org/project/embspec/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/MukundaKatta/embspec/actions/workflows/test.yml/badge.svg)](https://github.com/MukundaKatta/embspec/actions/workflows/test.yml)

Embedding pipeline ops + drift detection for production RAG.

The single failure mode this library prevents: query encoder upgrade ships before the index is re-encoded; every health check stays 200 OK while retrieval accuracy silently collapses. The [decompressed.io RAG observability post-mortem (2026-03-09)](https://decompressed.io/learn/rag-observability-postmortem) describes this exact bug — \$15K of emergency re-encoding plus 2-5 days of engineer time before someone diagnosed it.

```bash
pip install embspec
```

```python
from embspec import IndexManifest, embed_assert

manifest = IndexManifest.load("s3://my-rag/index-prod/manifest.json")

@embed_assert(manifest, model_id="amazon.titan-embed-text-v2:0", dimension=1024)
def search(query: str):
    qv = embed_query(query)
    return opensearch.knn_search(qv, ...)
```

If the query encoder ever drifts off the manifest, `search` raises `EmbeddingVersionMismatch` instead of silently returning bad results.

## What's in v0.1

| Primitive | What it does | Anchored in |
|---|---|---|
| `IndexManifest` + `EmbeddingSpec` | Single-file source of truth for "what does this index contain" — model id, dimension, version, normalization | [decompressed.io post-mortem](https://decompressed.io/learn/rag-observability-postmortem) |
| `assert_compatible()` / `embed_assert()` decorator | Fast-fail on encoder/index drift; raise or warn modes | same |
| `DriftAdapter` | Linear adapter from new-model embeddings to old-model space; lets you swap the query encoder without re-encoding the corpus | [Drift-Adapter, Vejendla 2025 (arxiv:2509.23471)](https://arxiv.org/abs/2509.23471) |
| `neighbor_stability()` | Compare two retrievers on a frozen probe set; reports overlap, Jaccard, regression list, deploy-safety verdict | [RAGOps survey, Xu et al. 2025 (arxiv:2506.03401)](https://arxiv.org/abs/2506.03401) |

## Why not Evidently / Phoenix / WhyLogs?

- **Evidently** — tabular-ML drift heritage; LLM additions are recent and platform-shaped. Not a drop-in primitive.
- **Phoenix** — embedding-drift visualization is a sub-feature of a full observability platform. You adopt the platform.
- **WhyLogs** — generic data-logging primitive; not embedding-aware; last commit 2025-01.
- **embspec** — three small primitives (`IndexManifest`, `DriftAdapter`, `neighbor_stability`) you compose with whatever vector DB and tracer you already have. No platform, no UI, no agent framework.

## Usage

### Manifest + version assertion

```python
from datetime import datetime, timezone
from embspec import IndexManifest, EmbeddingSpec

# When you build the index, write a manifest alongside it
manifest = IndexManifest(
    index_name="prod-v3",
    embedding=EmbeddingSpec(
        model_id="amazon.titan-embed-text-v2:0",
        dimension=1024,
        normalization="l2",
    ),
    created_at=datetime.now(timezone.utc),
    doc_count=8_000_000,
)
manifest.save("s3://my-rag/index-prod/manifest.json")
# (or any local path; manifest.save uses pathlib.Path.write_text)

# At query time, assert the encoder matches before searching
from embspec import embed_assert

@embed_assert(
    "s3://my-rag/index-prod/manifest.json",  # path or IndexManifest
    model_id="amazon.titan-embed-text-v2:0",
    dimension=1024,
    mode="raise",  # or "log" for canary rollout
)
def search(query: str) -> list[dict]:
    qv = embed_query(query)
    return opensearch.knn_search(qv, ...)
```

### Drift-Adapter for in-place model migration

When you want to upgrade the query encoder without re-encoding 8M docs:

```python
from embspec import DriftAdapter
import numpy as np

# Sample, e.g., 50K docs and embed them with both old and new models
old_emb = embed_with_old_model(sample_texts)  # shape (50000, 1024)
new_emb = embed_with_new_model(sample_texts)  # shape (50000, 1536)

adapter = DriftAdapter.fit(
    new_embeddings=new_emb,
    old_embeddings=old_emb,
    regularization=0.01,  # ridge; helps when new_emb is rank-deficient
)
adapter.save("s3://my-rag/adapters/v3-to-v4.npz")

# At query time, embed with the new model then transform into old space
qv_new = embed_with_new_model(query)
qv_compatible = adapter.transform(qv_new)  # shape (1024,)
results = opensearch.knn_search(qv_compatible, ...)
```

Per Vejendla 2025, this typically recovers 95-99% retrieval at ~1% of the cost of re-encoding the full corpus.

### Neighbor stability for safe migrations

```python
from embspec import neighbor_stability

# Run a fixed probe set against both indexes
old_results = {pid: retrieve_from_v3(q) for pid, q in probes.items()}
new_results = {pid: retrieve_from_v4(q) for pid, q in probes.items()}

report = neighbor_stability(old_results, new_results, k=10)
print(f"mean overlap@10: {report.mean_overlap_at_k:.1%}")
print(f"regressions: {report.regression_count}/{report.n_probes}")

if report.is_safe_to_deploy(min_mean_overlap=0.85, max_regression_fraction=0.05):
    deploy_v4()
else:
    investigate(report.regression_probe_ids)
```

## What it explicitly does NOT do

- Not a vector database.
- Not a RAG framework. No retriever, no chunker, no generator.
- Not a generic ML drift library. Embedding-and-retrieval-shaped only.
- Not an eval framework. `neighbor_stability` is the one judgment you can make without a labeled gold set; for richer evals use `ragas`, `trulens`, or a tracer.
- Does not host or serve embeddings.

## Roadmap

- v0.2: `dual_write()` context manager for blue/green index migrations across OpenSearch / pgvector / Pinecone / Qdrant.
- v0.3: `ChunkingExperiment` A/B harness with optional LLM-judge.
- v0.4: integration helpers for AWS Bedrock embedding models, OpenAI, Cohere, Voyage.

## License

Apache-2.0. See [LICENSE](./LICENSE).
