---
name: Bug report
about: Report a defect in embspec
title: ''
labels: bug
assignees: ''
---

## What happened

A clear, one-paragraph description of the bug.

## What I expected to happen

What you thought would happen.

## Repro

A minimal code sample that triggers the issue. For drift / stability bugs please include or describe the embedding-model + dimension involved.

```python
from embspec import IndexManifest, EmbeddingSpec

# ...
```

## Environment

- `embspec` version: (output of `pip show embspec`)
- Python version: (output of `python --version`)
- numpy version: (output of `pip show numpy`)
- Operating system:
- Embedding model + dimension involved (if any):
- Vector store backing the index (OpenSearch, pgvector, Pinecone, Qdrant, Chroma, Weaviate, none):

## Stack trace

If applicable, paste the full traceback.

```
```

## Anything else

Logs, manifest files (with any internal index names redacted), or related links.
