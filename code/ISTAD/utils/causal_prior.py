"""Training-only causal priors and entity-aware score calibration for ISTAD V7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class CausalPriorMetadata:
    n_points: int
    n_pairs: int
    n_features: int
    n_targets: int
    lag: int
    ridge: float
    topk: int
    entity_aware: bool
    coefficient_count: int
    prior_entropy_mean: float


def _validate_entity_ids(entity_ids, n_points):
    if entity_ids is None:
        return None
    entities = np.asarray(entity_ids).reshape(-1)
    if len(entities) != n_points:
        raise ValueError(
            f"entity_ids has {len(entities)} rows, expected {n_points}"
        )
    return entities


def fit_signed_causal_prior(
    points,
    target_indices: Sequence[int],
    *,
    lag: int = 1,
    ridge: float = 1e-2,
    topk: Optional[int] = None,
    entity_ids=None,
):
    """Fit a sparse signed source-to-target prior on normal training points.

    The design mirrors target-forecast inference.  At time ``t`` the selected
    target columns are replaced by their value at ``t-lag``; all remaining
    columns are contemporaneous exogenous context.  No pair crosses an entity
    boundary.  Absolute ridge coefficients define incidence probabilities and
    coefficient signs are retained for signed message aggregation.
    """

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError(f"points must have shape (time, features), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("points contain NaN or infinite values")
    n_points, n_features = values.shape
    targets = np.asarray(tuple(int(index) for index in target_indices), dtype=np.int64)
    if targets.size == 0 or np.unique(targets).size != targets.size:
        raise ValueError("target_indices must be non-empty and unique")
    if np.any(targets < 0) or np.any(targets >= n_features):
        raise ValueError("target_indices contains an out-of-range feature")
    lag = int(lag)
    ridge = float(ridge)
    if lag < 1 or lag >= n_points:
        raise ValueError("lag must lie in [1, n_points)")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")

    entities = _validate_entity_ids(entity_ids, n_points)
    valid = np.arange(lag, n_points, dtype=np.int64)
    if entities is not None:
        same_entity = np.ones(len(valid), dtype=bool)
        for offset in range(1, lag + 1):
            same_entity &= entities[valid] == entities[valid - offset]
        valid = valid[same_entity]
    if valid.size == 0:
        raise ValueError("no causal training pairs remain after entity filtering")

    design = values[valid].copy()
    design[:, targets] = values[valid - lag][:, targets]
    response = values[valid][:, targets]

    # An unpenalized intercept keeps entity/context offsets from being expressed
    # as spurious edges.  Only feature coefficients become graph structure.
    augmented = np.concatenate(
        [design, np.ones((len(design), 1), dtype=np.float64)], axis=1
    )
    gram = augmented.T @ augmented
    penalty = ridge * np.eye(gram.shape[0], dtype=np.float64)
    penalty[-1, -1] = 0.0
    cross = augmented.T @ response
    try:
        fitted = np.linalg.solve(gram + penalty, cross)
    except np.linalg.LinAlgError:
        fitted = np.linalg.lstsq(gram + penalty, cross, rcond=None)[0]
    coefficients = fitted[:-1]

    if topk is None or int(topk) <= 0:
        topk = max(3, int(np.ceil(0.2 * n_features)))
    topk = max(1, min(int(topk), n_features))
    importance = np.abs(coefficients)
    sparse = np.zeros_like(importance)
    strongest = np.argpartition(importance, -topk, axis=0)[-topk:]
    columns = np.arange(targets.size)[None, :]
    sparse[strongest, columns] = importance[strongest, columns]

    # Preserve the target's autoregressive edge even if numerical ties excluded
    # it, then provide a deterministic fallback for a constant target.
    sparse[targets, np.arange(targets.size)] = importance[
        targets, np.arange(targets.size)
    ]
    for column, target in enumerate(targets):
        if sparse[:, column].sum() <= 1e-12:
            sparse[target, column] = 1.0
    prior = sparse / np.maximum(sparse.sum(axis=0, keepdims=True), 1e-12)
    signs = np.sign(coefficients)
    signs[signs == 0.0] = 1.0

    entropy = -np.sum(
        np.where(prior > 0.0, prior * np.log(np.maximum(prior, 1e-12)), 0.0),
        axis=0,
    )
    metadata = CausalPriorMetadata(
        n_points=int(n_points),
        n_pairs=int(len(valid)),
        n_features=int(n_features),
        n_targets=int(targets.size),
        lag=lag,
        ridge=ridge,
        topk=topk,
        entity_aware=entities is not None,
        coefficient_count=int(coefficients.size),
        prior_entropy_mean=float(entropy.mean()),
    )
    return (
        prior.astype(np.float32),
        signs.astype(np.float32),
        coefficients.astype(np.float32),
        asdict(metadata),
    )


class EntitywiseComponentECDF:
    """Calibrate score components using only normal-training references."""

    def __init__(self):
        self.references_: Dict[int, list[np.ndarray]] = {}
        self.n_components_: Optional[int] = None
        self.entity_aware_: Optional[bool] = None

    @staticmethod
    def _validate_components(components):
        values = np.asarray(components, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or len(values) == 0:
            raise ValueError(
                f"components must have shape (points, components), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("components contain NaN or infinite values")
        return values

    def fit(self, train_components, entity_ids=None):
        values = self._validate_components(train_components)
        entities = _validate_entity_ids(entity_ids, len(values))
        if entities is None:
            entities = np.zeros(len(values), dtype=np.int64)
            self.entity_aware_ = False
        else:
            entities = entities.astype(np.int64, copy=False)
            self.entity_aware_ = True
        self.n_components_ = int(values.shape[1])
        self.references_ = {}
        for entity in np.unique(entities):
            subset = values[entities == entity]
            if len(subset) == 0:
                continue
            self.references_[int(entity)] = [
                np.sort(subset[:, component])
                for component in range(self.n_components_)
            ]
        if not self.references_:
            raise ValueError("no calibration references were fitted")
        return self

    def transform(self, components, entity_ids=None):
        if self.n_components_ is None:
            raise RuntimeError("fit() must be called before transform()")
        values = self._validate_components(components)
        if values.shape[1] != self.n_components_:
            raise ValueError(
                f"expected {self.n_components_} components, got {values.shape[1]}"
            )
        entities = _validate_entity_ids(entity_ids, len(values))
        if self.entity_aware_:
            if entities is None:
                raise ValueError("entity-aware calibration requires entity_ids")
            entities = entities.astype(np.int64, copy=False)
        else:
            entities = np.zeros(len(values), dtype=np.int64)

        calibrated = np.empty_like(values, dtype=np.float64)
        for entity in np.unique(entities):
            key = int(entity)
            if key not in self.references_:
                raise ValueError(f"entity {key} was absent from training calibration")
            keep = entities == entity
            for component, reference in enumerate(self.references_[key]):
                rank = np.searchsorted(
                    reference, values[keep, component], side="right"
                )
                calibrated[keep, component] = (
                    rank + 0.5
                ) / float(len(reference) + 1)
        return calibrated

    def metadata_dict(self):
        if self.n_components_ is None:
            raise RuntimeError("fit() must be called before reading metadata")
        return {
            "entity_aware": bool(self.entity_aware_),
            "n_components": int(self.n_components_),
            "reference_sizes": {
                str(entity): int(len(references[0]))
                for entity, references in self.references_.items()
            },
        }
