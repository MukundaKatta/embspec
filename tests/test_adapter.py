"""DriftAdapter tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from embspec import AdapterShapeError, DriftAdapter


def test_fit_perfectly_recovers_known_linear_map() -> None:
    rng = np.random.default_rng(0)
    n, old_dim, new_dim = 200, 64, 96
    true_W = rng.standard_normal((new_dim, old_dim))
    new_X = rng.standard_normal((n, new_dim))
    old_X = new_X @ true_W

    adapter = DriftAdapter.fit(new_embeddings=new_X, old_embeddings=old_X)
    np.testing.assert_allclose(adapter.weight, true_W, atol=1e-6)
    assert adapter.input_dim == new_dim
    assert adapter.output_dim == old_dim


def test_transform_2d_input() -> None:
    rng = np.random.default_rng(1)
    new_X = rng.standard_normal((50, 32))
    old_X = rng.standard_normal((50, 16))
    adapter = DriftAdapter.fit(new_embeddings=new_X, old_embeddings=old_X)
    out = adapter.transform(new_X[:5])
    assert out.shape == (5, 16)


def test_transform_1d_input_returns_1d() -> None:
    rng = np.random.default_rng(2)
    new_X = rng.standard_normal((50, 32))
    old_X = rng.standard_normal((50, 16))
    adapter = DriftAdapter.fit(new_embeddings=new_X, old_embeddings=old_X)
    out = adapter.transform(new_X[0])
    assert out.shape == (16,)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    new_X = rng.standard_normal((100, 16))
    old_X = rng.standard_normal((100, 8))
    adapter = DriftAdapter.fit(new_embeddings=new_X, old_embeddings=old_X)
    path = tmp_path / "adapter.npz"
    adapter.save(path)
    loaded = DriftAdapter.load(path)
    np.testing.assert_array_equal(adapter.weight, loaded.weight)
    np.testing.assert_allclose(
        adapter.transform(new_X[:5]), loaded.transform(new_X[:5])
    )


def test_fit_raises_on_row_count_mismatch() -> None:
    new_X = np.zeros((50, 16))
    old_X = np.zeros((49, 8))
    with pytest.raises(AdapterShapeError) as exc_info:
        DriftAdapter.fit(new_embeddings=new_X, old_embeddings=old_X)
    assert "50" in str(exc_info.value) and "49" in str(exc_info.value)


def test_fit_raises_when_too_few_samples_for_dimensions() -> None:
    new_X = np.zeros((10, 16))
    old_X = np.zeros((10, 8))
    with pytest.raises(AdapterShapeError):
        DriftAdapter.fit(new_embeddings=new_X, old_embeddings=old_X)


def test_transform_raises_on_dim_mismatch() -> None:
    rng = np.random.default_rng(4)
    new_X = rng.standard_normal((50, 32))
    old_X = rng.standard_normal((50, 16))
    adapter = DriftAdapter.fit(new_embeddings=new_X, old_embeddings=old_X)
    with pytest.raises(AdapterShapeError):
        adapter.transform(rng.standard_normal((5, 64)))


def test_ridge_regularization_path_runs() -> None:
    rng = np.random.default_rng(5)
    new_X = rng.standard_normal((100, 16))
    old_X = rng.standard_normal((100, 8))
    adapter = DriftAdapter.fit(
        new_embeddings=new_X, old_embeddings=old_X, regularization=0.5
    )
    assert adapter.weight.shape == (16, 8)


def test_init_rejects_non_2d_weight() -> None:
    with pytest.raises(AdapterShapeError):
        DriftAdapter(weight=np.zeros(8))


def test_drift_adapter_recovers_high_cosine_on_synthetic_drift() -> None:
    """Sanity check: when we simulate model drift via a small perturbation,
    the adapter recovers cosine similarity close to identity."""
    rng = np.random.default_rng(6)
    n, d = 500, 64
    old_X = rng.standard_normal((n, d))
    # Simulated new model: same space + small noise
    drift_W = np.eye(d) + 0.05 * rng.standard_normal((d, d))
    new_X = old_X @ drift_W + 0.01 * rng.standard_normal((n, d))

    adapter = DriftAdapter.fit(
        new_embeddings=new_X[:300], old_embeddings=old_X[:300]
    )

    # On held-out probes, transformed new should be near old
    held_new = new_X[300:]
    held_old = old_X[300:]
    recovered = adapter.transform(held_new)

    # Mean cosine similarity per row
    cos = (recovered * held_old).sum(axis=1) / (
        np.linalg.norm(recovered, axis=1) * np.linalg.norm(held_old, axis=1)
    )
    assert cos.mean() > 0.95
