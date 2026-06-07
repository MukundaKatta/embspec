# Changelog

All notable changes to `embspec` are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `IndexManifest.to_dict()` no longer silently shifts a naive (tz-unaware) `created_at` by the host machine's local UTC offset. A naive datetime is now treated as UTC, so the same manifest serializes identically regardless of which timezone the build machine is in. Aware datetimes continue to be normalized to UTC.
- `IndexManifest.from_dict()` / `IndexManifest.load()` now raise the library's typed `ManifestFormatError` (instead of leaking a raw `KeyError`/`ValueError`/`TypeError`) when a manifest's `embedding` block is missing `model_id`/`dimension`, has a non-numeric `dimension`, is not an object, or carries an unparseable `created_at`. Callers can now catch a single exception for any malformed manifest.

### Added
- Standard-library `unittest` test suite (`tests/test_*_unittest.py`) covering the numpy-free primitives (`IndexManifest`, `assert_compatible`, `embed_assert`, `neighbor_stability`) plus regression tests for the fixes above. Runnable with `python3 -m unittest discover -s tests` even when `numpy`/`pytest` are not installed; the numpy/pytest-dependent modules skip cleanly in that environment.

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
