# Changelog

All notable changes to `embspec` are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-08

Initial release. Three primitives that close the embedding-pipeline-ops gap left between Evidently (tabular-ML drift), Phoenix (full observability platform), and the Drift-Adapter paper (research code).

### Added
- `IndexManifest` and `EmbeddingSpec` dataclasses with JSON load/save.
- `assert_compatible()` and `embed_assert()` decorator with `raise` and `log` modes; raises `EmbeddingVersionMismatch` on encoder drift. Fixes the silent-accuracy-collapse failure mode described in the decompressed.io RAG observability post-mortem (2026-03-09).
- `DriftAdapter`: linear least-squares fit + transform + `.npz` save/load. Implements Vejendla 2025 ([arxiv:2509.23471](https://arxiv.org/abs/2509.23471)) — recovers 95-99% retrieval after embedding model swap without re-encoding the corpus. Supports ridge regularization.
- `neighbor_stability()`: pure function comparing two retrievers on a frozen probe set; returns `StabilityReport` with mean overlap@k, mean Jaccard@k, regression probe ids, and a `is_safe_to_deploy()` heuristic gate. Targets the retrieval-side observability gap called out in RAGOps (Xu et al. 2025, [arxiv:2506.03401](https://arxiv.org/abs/2506.03401)).
- Typed errors: `EmbspecError`, `EmbeddingVersionMismatch`, `ManifestFormatError`, `AdapterShapeError`.

### Notes
- 32 unit tests, 99% line coverage, deterministic numpy seeds throughout.
- Lazy `numpy` imports in `_adapter.py` so `import embspec` is cheap when only the manifest+assert APIs are needed.
- Format version field (`embspec_format_version: 1`) on every manifest enables forward-compatible upgrades.
