# Porting status

Where embspec is published, where it's queued, and what's intentionally not happening.

## Current

| Channel | State | URL |
|---|---|---|
| GitHub Release | live | https://github.com/MukundaKatta/embspec/releases/tag/v0.1.0 |
| PyPI | live | https://pypi.org/project/embspec/0.1.0/ |
| conda-forge | recipe PR submitted | https://github.com/conda-forge/staged-recipes/pull/33282 |
| nixpkgs | derivation PR submitted | https://github.com/NixOS/nixpkgs/pull/518492 |
| Homebrew tap | formula PR submitted | https://github.com/MukundaKatta/homebrew-tools/pull/1 |
| MCP server | live on npm | https://www.npmjs.com/package/@mukundakatta/embspec-mcp |

Install:

```bash
pip install embspec
```

## Roadmap

### Plausible

| Target | Why | Approx scope |
|---|---|---|
| **Go port** | Awkward but doable via gonum for the linear algebra; useful for Go-native RAG services | ~1 week |

### Skipped — TypeScript port

A TS port was considered. Skipped for v0.1 because:

- numpy-equivalents in JS (math.js, ndarray) are awkward and slow
- The bulk of RAG production work is Python; TS-RAG is a smaller audience for drift work specifically
- The IndexManifest pattern is language-agnostic; if needed, a thin TS lib could just do the manifest assertion piece without porting the DriftAdapter math

If there's specific demand for a TS port, file an issue with the use case.

### Not planned

| Target | Why not |
|---|---|
| Java / Ruby / PHP / Perl / Haskell / OCaml / Swift | No real RAG-on-X community for these ecosystems |
| Conan / vcpkg | C/C++ only |
| Homebrew / APT / DNF / Pacman / Chocolatey | Library-only formulas rejected; distro packages target apps |
| Docker Hub / GHCR / Quay | Libraries aren't containers |

## How to contribute a port

Open an issue with the target ecosystem and a sketch of the public-API mapping before writing code.
