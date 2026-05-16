## Summary

What does this PR change in 1-3 sentences?

## Linked issue

Closes #

## Scope check

- [ ] Confirmed this fits `CONTRIBUTING.md` scope (embedding-pipeline-ops + drift detection, not a vector DB or RAG framework).
- [ ] Public API changes (if any) are documented in the README and CHANGELOG.
- [ ] No new runtime dependencies added (or, if added, justified in the description).
- [ ] If changing `IndexManifest` JSON shape, bumped `embspec_format_version` and added a backward-compatible loader.

## Tests

- [ ] Added or updated tests covering the behavior change.
- [ ] `uv run pytest` passes locally.
- [ ] `uv run pytest --cov=embspec` coverage at or above 95%.
- [ ] Any numpy-using paths have deterministic seeded test data.

## Risk and impact

Anything reviewers should look at extra carefully (numpy version sensitivity, large-array memory, manifest format compatibility, etc.)?

## Notes for the reviewer

Anything off-checklist worth surfacing.
