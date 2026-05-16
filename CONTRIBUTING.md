# Contributing to embspec

embspec is a focused library for the embedding-pipeline-ops + drift-detection corner of production RAG. Contributions are welcome where they fit that scope; out-of-scope contributions will be politely declined.

## In scope

- Bug fixes in `IndexManifest`, `EmbeddingSpec`, `assert_compatible`, `embed_assert`, `DriftAdapter`, `neighbor_stability`.
- Additional manifest-format-version migrations (when we bump `embspec_format_version`).
- Improvements to `DriftAdapter` (alternative fit methods, support for sparse embeddings, etc.) — but only as additive options, not replacements.
- New per-element stability metrics in `neighbor_stability` (e.g. NDCG-style relevance-weighted overlap), behind keyword arguments.
- Test coverage improvements (current target: 95%+ line coverage).
- Documentation fixes.

## Out of scope

- Vector databases or vector store integrations. embspec is index-agnostic by design; you pass in retrieval results, you don't store vectors here.
- RAG retrieval frameworks (LangChain, LlamaIndex). embspec can be used inside any of them but doesn't depend on them.
- Embedding model wrappers. embspec compares results; it doesn't call models.
- Generic ML drift detection (tabular features, concept drift). embspec is embedding-and-retrieval-shaped only.

## Development setup

```bash
git clone https://github.com/MukundaKatta/embspec.git
cd embspec
uv sync --group dev
uv run pytest                                # 32 tests, 99% coverage
uv run pytest --cov=embspec --cov-report=term-missing
uv build                                     # build sdist + wheel
```

Python 3.10+ required. numpy is the only runtime dependency.

## Workflow

1. Open an issue first for anything bigger than a one-file change. This avoids spending hours on something that turns out to be out-of-scope.
2. Branch from `main`.
3. Write tests before or alongside the change. `DriftAdapter` in particular must keep deterministic seeded test paths.
4. Run `uv run pytest` and confirm full suite still passes.
5. Open a PR against `main`. Fill in the template. Link the issue.
6. CI must be green before review.

## Coding conventions

- Type hints required on public APIs.
- numpy arrays should have asserted shapes in fit/transform paths so dimensional errors surface with a clear message.
- Keep public symbols in `__all__`; otherwise they aren't re-exported.
- `IndexManifest` JSON format changes require a version bump (`embspec_format_version`) and a backward-compatible loader.

## Release cadence

Releases follow semver. Patches: bug fixes only. Minor versions: new public symbols. Major versions: breaking changes (unlikely in v0.x).

Releases are cut by the maintainer via tag push. See `.github/workflows/release.yml`.
