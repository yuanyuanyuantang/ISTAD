"""Training-only temporal innovation scoring for ISTAD.

The scorer is deliberately small: it fits a ridge-regularized vector
autoregression on normal training points, then measures one-step innovations.
No test values or labels are used to fit its coefficients, feature scales, pool
selection, or empirical calibration reference.

Two pooling regimes are useful for multivariate telemetry:

* ``sparse``: maximum standardized feature innovation, for datasets where a
  small subset of sensors changes abnormally;
* ``dense``: mean squared innovation, used when many training channels are
  constant.  Standardizing such channels would otherwise amplify numerical
  noise and discrete state changes.

``auto`` chooses between them from the normal-training residuals alone.
"""

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


def hypergraph_pool_feature_evidence(feature_evidence, incidence, pool="sparse"):
    """Route causal innovation evidence through a dynamic hypergraph.

    ``incidence[..., node, edge]`` is expected to be normalized over nodes for
    every hyperedge, as produced by :class:`LightweightHypergraphMixer`.  The
    operation is deliberately parameter free: HGAT learns *which variables
    belong together*, while the closed-form V4 branch supplies the anomaly
    evidence.  This keeps the final detector auditable and prevents a second
    neural score head from quietly replacing V4.
    """
    evidence = np.asarray(feature_evidence, dtype=np.float64)
    relation = np.asarray(incidence, dtype=np.float64)
    if evidence.ndim != 2:
        raise ValueError(
            f"feature_evidence must have shape (time, nodes), got {evidence.shape}"
        )
    if relation.ndim != 3:
        raise ValueError(
            f"incidence must have shape (time, nodes, edges), got {relation.shape}"
        )
    if relation.shape[:2] != evidence.shape:
        raise ValueError(
            "feature evidence/incidence mismatch: "
            f"{evidence.shape} versus {relation.shape}"
        )
    if pool not in {"dense", "sparse"}:
        raise ValueError("pool must be 'dense' or 'sparse'")
    if not np.isfinite(evidence).all() or not np.isfinite(relation).all():
        raise ValueError("feature evidence and incidence must be finite")

    mass = relation.sum(axis=1, keepdims=True)
    normalized = relation / np.maximum(mass, np.finfo(np.float64).eps)
    edge_evidence = np.einsum("tnm,tn->tm", normalized, evidence)
    if pool == "dense":
        return edge_evidence.mean(axis=1)
    return edge_evidence.max(axis=1)


def training_only_hgat_reliability_gate(
    base_score,
    graph_score,
    entity_ids=None,
    max_weight=0.20,
    reference_fraction=0.80,
    tail_probability=0.01,
    inflation_limit=1.25,
):
    """Choose an HGAT fusion weight without anomaly labels or test values.

    The first part of each normal-training entity defines a tail threshold and
    the remaining part measures false-alarm drift.  HGAT is disabled when its
    normal-tail rate is materially worse than both the requested tail rate and
    the V4 branch.  Otherwise its fixed maximum contribution is scaled by its
    stability relative to V4.  A zero gate is an exact V4 kill-switch.
    """
    base = np.asarray(base_score, dtype=np.float64).reshape(-1)
    graph = np.asarray(graph_score, dtype=np.float64).reshape(-1)
    if len(base) != len(graph) or len(base) < 4:
        raise ValueError("base_score and graph_score need the same non-trivial length")
    if not np.isfinite(base).all() or not np.isfinite(graph).all():
        raise ValueError("training scores must be finite")
    if not 0.0 <= max_weight <= 1.0:
        raise ValueError("max_weight must lie in [0, 1]")
    if not 0.0 < reference_fraction < 1.0:
        raise ValueError("reference_fraction must lie in (0, 1)")
    if not 0.0 < tail_probability < 0.5:
        raise ValueError("tail_probability must lie in (0, 0.5)")
    if inflation_limit < 1.0:
        raise ValueError("inflation_limit must be at least 1")

    if entity_ids is None:
        entities = np.zeros(len(base), dtype=np.int64)
    else:
        entities = np.asarray(entity_ids).reshape(-1)
        if len(entities) != len(base):
            raise ValueError("entity_ids must align with the training scores")

    reference_mask = np.zeros(len(base), dtype=bool)
    validation_mask = np.zeros(len(base), dtype=bool)
    for entity in np.unique(entities):
        indices = np.flatnonzero(entities == entity)
        split = min(len(indices) - 1, max(1, int(len(indices) * reference_fraction)))
        reference_mask[indices[:split]] = True
        validation_mask[indices[split:]] = True
    if not reference_mask.any() or not validation_mask.any():
        raise ValueError("normal-training reliability split is empty")

    quantile = 1.0 - tail_probability

    def tail_rate(values):
        threshold = float(np.quantile(values[reference_mask], quantile))
        return float(np.mean(values[validation_mask] > threshold)), threshold

    base_rate, base_threshold = tail_rate(base)
    graph_rate, graph_threshold = tail_rate(graph)
    allowed_rate = max(
        inflation_limit * tail_probability,
        inflation_limit * base_rate,
    )
    killed = bool(graph_rate > allowed_rate)
    if killed or max_weight == 0.0:
        reliability = 0.0
        weight = 0.0
    else:
        base_error = abs(base_rate - tail_probability)
        graph_error = abs(graph_rate - tail_probability)
        reliability = min(
            1.0,
            (tail_probability + base_error)
            / (tail_probability + graph_error + np.finfo(np.float64).eps),
        )
        weight = float(max_weight * reliability)

    metadata = {
        "method": "normal_tail_stability",
        "reference_fraction": float(reference_fraction),
        "tail_probability": float(tail_probability),
        "inflation_limit": float(inflation_limit),
        "reference_size": int(reference_mask.sum()),
        "validation_size": int(validation_mask.sum()),
        "base_threshold": base_threshold,
        "graph_threshold": graph_threshold,
        "base_validation_tail_rate": base_rate,
        "graph_validation_tail_rate": graph_rate,
        "allowed_graph_tail_rate": float(allowed_rate),
        "reliability": float(reliability),
        "max_weight": float(max_weight),
        "selected_weight": weight,
        "kill_switch": killed,
    }
    return weight, metadata


def rank_safe_hgat_refine(base_score, graph_score, calibration_size):
    """Use bounded HGAT evidence only below one empirical V4 rank step."""
    base = np.asarray(base_score, dtype=np.float64)
    graph = np.asarray(graph_score, dtype=np.float64)
    if base.shape != graph.shape:
        raise ValueError("base_score and graph_score must have matching shapes")
    if int(calibration_size) < 1:
        raise ValueError("calibration_size must be positive")
    if not np.isfinite(base).all() or not np.isfinite(graph).all():
        raise ValueError("scores must be finite")
    if np.any(graph < 0.0) or np.any(graph > 1.0):
        raise ValueError("graph_score must be bounded in [0, 1]")
    epsilon = 0.5 / float(int(calibration_size) + 1)
    refined = (base + epsilon * graph) / (1.0 + epsilon)
    return refined, epsilon


def training_ecdf_recalibrate(train_score, *score_arrays):
    """Map fused scores through a right-sided, training-only empirical CDF.

    The returned first array is the recalibrated training score.  Every other
    array uses exactly the same sorted normal-training reference, so evaluation
    values cannot influence the mapping.  This is useful after mixing two
    individually calibrated branches: their convex combination is generally no
    longer calibrated on the normal training distribution.
    """
    train = np.asarray(train_score, dtype=np.float64)
    original_shape = train.shape
    train_flat = train.reshape(-1)
    if train_flat.size == 0:
        raise ValueError("train_score must not be empty")
    if not np.isfinite(train_flat).all():
        raise ValueError("train_score contains NaN or infinite values")

    reference = np.sort(train_flat)

    def transform(values):
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("score array contains NaN or infinite values")
        rank = np.searchsorted(reference, array.reshape(-1), side="right")
        return (rank / float(len(reference))).reshape(array.shape)

    calibrated = [transform(train).reshape(original_shape)]
    calibrated.extend(transform(values) for values in score_arrays)
    return tuple(calibrated)


@dataclass(frozen=True)
class InnovationMetadata:
    lag: int
    ridge: float
    n_features: int
    coefficient_count: int
    requested_pool: str
    selected_pool: str
    degenerate_fraction: float
    degenerate_cutoff: float
    scale_floor_ratio: float
    calibration_size: int


class TrainingOnlyInnovationScorer:
    """Closed-form causal residual branch with train-only ECDF calibration."""

    VALID_POOLS = {"auto", "dense", "sparse"}

    def __init__(
        self,
        lag: int = 1,
        ridge: float = 1e-2,
        pool: str = "auto",
        degenerate_cutoff: float = 0.10,
        scale_floor_ratio: float = 0.10,
        zero_scale_epsilon: float = 1e-8,
    ):
        self.lag = int(lag)
        self.ridge = float(ridge)
        self.pool = str(pool).lower()
        self.degenerate_cutoff = float(degenerate_cutoff)
        self.scale_floor_ratio = float(scale_floor_ratio)
        self.zero_scale_epsilon = float(zero_scale_epsilon)
        if self.lag < 1:
            raise ValueError("lag must be at least 1")
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")
        if self.pool not in self.VALID_POOLS:
            raise ValueError(f"pool must be one of {sorted(self.VALID_POOLS)}")
        if not 0.0 <= self.degenerate_cutoff <= 1.0:
            raise ValueError("degenerate_cutoff must lie in [0, 1]")
        if self.scale_floor_ratio <= 0:
            raise ValueError("scale_floor_ratio must be positive")

        self.coef_: Optional[np.ndarray] = None
        self.residual_scale_: Optional[np.ndarray] = None
        self.reference_: Optional[np.ndarray] = None
        self.selected_pool_: Optional[str] = None
        self.degenerate_fraction_: Optional[float] = None
        self.n_features_: Optional[int] = None

    @staticmethod
    def _validate_points(points, entity_ids=None):
        values = np.asarray(points)
        if values.ndim != 2:
            raise ValueError(f"points must have shape (time, features), got {values.shape}")
        if len(values) == 0:
            raise ValueError("points must not be empty")
        if not np.isfinite(values).all():
            raise ValueError("points contain NaN or infinite values")
        entities = None
        if entity_ids is not None:
            entities = np.asarray(entity_ids).reshape(-1)
            if len(entities) != len(values):
                raise ValueError(
                    f"entity_ids has {len(entities)} rows, expected {len(values)}"
                )
        return values, entities

    def _lagged(self, points, entity_ids=None):
        values, entities = self._validate_points(points, entity_ids)
        if len(values) <= self.lag:
            raise ValueError(
                f"need more than lag={self.lag} points, got {len(values)}"
            )

        valid = np.arange(self.lag, len(values), dtype=np.int64)
        if entities is not None:
            same_entity = np.ones(len(valid), dtype=bool)
            for offset in range(1, self.lag + 1):
                same_entity &= entities[valid] == entities[valid - offset]
            valid = valid[same_entity]
        if len(valid) == 0:
            raise ValueError("no valid causal pairs remain after applying entity boundaries")

        design = np.concatenate(
            [values[valid - offset] for offset in range(1, self.lag + 1)],
            axis=1,
        )
        return valid, design, values[valid]

    def _residuals(self, points, entity_ids=None):
        if self.coef_ is None:
            raise RuntimeError("fit() must be called before scoring")
        valid, design, target = self._lagged(points, entity_ids)
        residual = target.astype(np.float64, copy=False) - (
            design.astype(np.float64, copy=False) @ self.coef_
        )
        return valid, residual

    def _pool_residuals(self, residual):
        if self.selected_pool_ == "dense":
            return np.mean(np.square(residual), axis=1)
        if self.selected_pool_ == "sparse":
            return np.max(
                np.abs(residual) / self.residual_scale_[None, :], axis=1
            )
        raise RuntimeError("fit() must select a pooling mode before scoring")

    def fit(self, points, entity_ids=None):
        values, entities = self._validate_points(points, entity_ids)
        valid, design, target = self._lagged(values, entities)
        design64 = design.astype(np.float64, copy=False)
        target64 = target.astype(np.float64, copy=False)
        gram = design64.T @ design64
        cross = design64.T @ target64
        regularized = gram + self.ridge * np.eye(gram.shape[0], dtype=np.float64)
        try:
            self.coef_ = np.linalg.solve(regularized, cross)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(regularized, cross, rcond=None)[0]

        residual = target64 - design64 @ self.coef_
        raw_scale = np.std(residual, axis=0)
        positive_scale = raw_scale[raw_scale > self.zero_scale_epsilon]
        if len(positive_scale) == 0:
            raise ValueError("all innovation channels are constant on the training data")
        median_scale = float(np.median(positive_scale))
        self.residual_scale_ = np.maximum(
            raw_scale, self.scale_floor_ratio * median_scale
        )
        self.degenerate_fraction_ = float(
            np.mean(raw_scale <= self.zero_scale_epsilon)
        )
        self.n_features_ = int(values.shape[1])
        self.selected_pool_ = self.pool
        if self.pool == "auto":
            self.selected_pool_ = (
                "dense"
                if self.degenerate_fraction_ >= self.degenerate_cutoff
                else "sparse"
            )

        reference = self._pool_residuals(residual)
        self.reference_ = np.sort(np.asarray(reference, dtype=np.float64))
        if not np.isfinite(self.reference_).all():
            raise RuntimeError("non-finite innovation calibration reference")
        return self

    def raw_score(self, points, entity_ids=None):
        values, entities = self._validate_points(points, entity_ids)
        valid, residual = self._residuals(values, entities)
        score = np.zeros(len(values), dtype=np.float64)
        score[valid] = self._pool_residuals(residual)
        return score

    def feature_evidence(self, points, entity_ids=None):
        """Return aligned per-variable evidence before V4 node pooling.

        Sparse datasets retain the standardized absolute innovation used by
        V4's maximum pool.  Dense/degenerate datasets retain the unstandardized
        squared innovation used by V4's mean pool.  Pooling this output over
        nodes therefore exactly reconstructs :meth:`raw_score`.
        """
        values, entities = self._validate_points(points, entity_ids)
        valid, residual = self._residuals(values, entities)
        evidence = np.zeros_like(values, dtype=np.float64)
        if self.selected_pool_ == "dense":
            evidence[valid] = np.square(residual)
        elif self.selected_pool_ == "sparse":
            evidence[valid] = (
                np.abs(residual) / self.residual_scale_[None, :]
            )
        else:
            raise RuntimeError("fit() must select a pooling mode before scoring")
        return evidence

    def signed_feature_innovation(self, points, entity_ids=None):
        """Return aligned, train-scaled signed innovations for relation learning.

        Unlike :meth:`feature_evidence`, this representation retains residual
        direction.  Its coefficients and feature scales are still fitted only
        on normal training points, and invalid lag prefixes remain zero.
        """
        values, entities = self._validate_points(points, entity_ids)
        valid, residual = self._residuals(values, entities)
        innovation = np.zeros_like(values, dtype=np.float64)
        innovation[valid] = residual / self.residual_scale_[None, :]
        return innovation

    def score(self, points, entity_ids=None, tail_transform=False):
        """Return a train-ECDF score, optionally mapped to an EVT-friendly tail.

        The half-rank correction keeps all values strictly inside ``(0, 1)``;
        consequently ``-log(1-u)`` remains finite for POT/SPOT evaluation.
        """
        if self.reference_ is None:
            raise RuntimeError("fit() must be called before scoring")
        raw = self.raw_score(points, entity_ids)
        rank = np.searchsorted(self.reference_, raw, side="right")
        calibrated = (rank + 0.5) / float(len(self.reference_) + 1)
        if tail_transform:
            return -np.log1p(-calibrated)
        return calibrated

    @property
    def metadata(self):
        if self.reference_ is None:
            raise RuntimeError("fit() must be called before reading metadata")
        return InnovationMetadata(
            lag=self.lag,
            ridge=self.ridge,
            n_features=self.n_features_,
            coefficient_count=int(self.coef_.size),
            requested_pool=self.pool,
            selected_pool=self.selected_pool_,
            degenerate_fraction=self.degenerate_fraction_,
            degenerate_cutoff=self.degenerate_cutoff,
            scale_floor_ratio=self.scale_floor_ratio,
            calibration_size=int(len(self.reference_)),
        )

    def metadata_dict(self):
        return asdict(self.metadata)
