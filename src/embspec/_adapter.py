"""Drift-Adapter: linear map from new-model embeddings into old-model embedding space.

Implements the pattern from Vejendla 2025 (arxiv:2509.23471, "Drift-Adapter:
Closing the Embedding-Drift Gap in Production Vector Stores"). Lets you swap
the query encoder to a newer embedding model without re-encoding the
corpus, by fitting a small linear transform on a sample of paired
(old_model, new_model) embeddings.

Trade-off: the adapter is a least-squares fit so it loses some signal vs.
true re-encoding. The paper reports 95-99% retrieval recovery with adapter
sizes ~1% of the corpus. Use it as a cost-saving migration path, then
re-encode the corpus on a slower schedule (or never if recall holds).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ._errors import AdapterShapeError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class DriftAdapter:
    """Linear adapter that maps embeddings from one model into another's space.

    Fitting solves ``W = argmin || X_new @ W - X_old ||_F^2`` where
    ``X_old`` and ``X_new`` are aligned matrices of paired embeddings.
    """

    def __init__(self, weight: NDArray) -> None:
        if weight.ndim != 2:
            raise AdapterShapeError(f"weight must be 2-D, got shape {weight.shape}")
        self.weight = weight

    @property
    def input_dim(self) -> int:
        return int(self.weight.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.weight.shape[1])

    @classmethod
    def fit(
        cls,
        new_embeddings: NDArray,
        old_embeddings: NDArray,
        *,
        regularization: float = 0.0,
    ) -> DriftAdapter:
        """Fit the adapter on aligned (new, old) embedding pairs via least squares.

        ``new_embeddings`` and ``old_embeddings`` must have the same number
        of rows (``n``) and may have different column counts (the new and
        old model dimensions). ``regularization`` adds an L2 penalty on the
        weight matrix (ridge regression); useful for ill-conditioned data.
        """
        import numpy as np  # noqa: PLC0415

        if new_embeddings.ndim != 2 or old_embeddings.ndim != 2:
            raise AdapterShapeError(
                f"embedding arrays must be 2-D, got {new_embeddings.shape} and {old_embeddings.shape}"
            )
        if new_embeddings.shape[0] != old_embeddings.shape[0]:
            raise AdapterShapeError(
                f"row counts differ: new={new_embeddings.shape[0]}, old={old_embeddings.shape[0]}; "
                "embeddings must be paired"
            )
        if new_embeddings.shape[0] < new_embeddings.shape[1]:
            raise AdapterShapeError(
                f"need at least as many paired samples as new-model dimensions; "
                f"got {new_embeddings.shape[0]} samples for {new_embeddings.shape[1]} dimensions"
            )

        if regularization > 0:
            n_features = new_embeddings.shape[1]
            xtx = new_embeddings.T @ new_embeddings + regularization * np.eye(n_features)
            xty = new_embeddings.T @ old_embeddings
            weight = np.linalg.solve(xtx, xty)
        else:
            weight, _, _, _ = np.linalg.lstsq(
                new_embeddings, old_embeddings, rcond=None
            )
        return cls(weight=weight)

    def transform(self, new_embeddings: NDArray) -> NDArray:
        """Map new-model embeddings into the old-model embedding space."""
        if new_embeddings.ndim == 1:
            if new_embeddings.shape[0] != self.input_dim:
                raise AdapterShapeError(
                    f"expected vector of dim {self.input_dim}, got {new_embeddings.shape[0]}"
                )
            return new_embeddings @ self.weight
        if new_embeddings.ndim != 2 or new_embeddings.shape[1] != self.input_dim:
            raise AdapterShapeError(
                f"expected (n, {self.input_dim}) input, got {new_embeddings.shape}"
            )
        return new_embeddings @ self.weight

    def save(self, path: str | Path) -> None:
        """Save the adapter to a compressed ``.npz`` file."""
        import numpy as np  # noqa: PLC0415

        np.savez_compressed(str(path), weight=self.weight)

    @classmethod
    def load(cls, path: str | Path) -> DriftAdapter:
        """Load an adapter previously saved with :meth:`save`."""
        import numpy as np  # noqa: PLC0415

        data = np.load(str(path))
        return cls(weight=data["weight"])
