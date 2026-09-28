"""Entity-safe event metrics for time-series anomaly detection.

An event is a contiguous positive run within one entity.  A predicted event is
correct when it overlaps at least one ground-truth event; a ground-truth event
is detected when at least one prediction overlaps it.  This segment-level F1
does not inflate a hit to the full event duration and therefore remains
distinct from point adjustment.
"""

from __future__ import annotations

import numpy as np


def _validated_binary(values, name):
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isin(array, (0, 1, False, True)).all():
        raise ValueError(f"{name} must be binary")
    return array.astype(bool, copy=False)


def event_ranges(values, entity_ids=None):
    """Return half-open ``(start, end)`` positive runs without crossing entities."""
    positive = _validated_binary(values, "values")
    if entity_ids is None:
        entities = np.zeros(len(positive), dtype=np.int64)
    else:
        entities = np.asarray(entity_ids).reshape(-1)
        if len(entities) != len(positive):
            raise ValueError("entity_ids must align with values")

    boundary_before = np.r_[True, entities[1:] != entities[:-1]]
    boundary_after = np.r_[entities[1:] != entities[:-1], True]
    starts = np.flatnonzero(positive & (boundary_before | np.r_[True, ~positive[:-1]]))
    ends = np.flatnonzero(positive & (boundary_after | np.r_[~positive[1:], True])) + 1
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def event_overlap_metrics(label, prediction, entity_ids=None):
    """Compute overlap event precision/recall/F1 and detection delay.

    Delay is measured from a true event's first point to its first positive
    prediction.  It is averaged only over detected events; ``None`` is returned
    when no event is detected.
    """
    actual = _validated_binary(label, "label")
    predicted = _validated_binary(prediction, "prediction")
    if len(actual) != len(predicted):
        raise ValueError("label and prediction must have equal length")
    if entity_ids is not None and len(np.asarray(entity_ids).reshape(-1)) != len(actual):
        raise ValueError("entity_ids must align with label")

    true_events = event_ranges(actual, entity_ids)
    predicted_events = event_ranges(predicted, entity_ids)
    true_hits = [bool(predicted[start:end].any()) for start, end in true_events]
    predicted_hits = [bool(actual[start:end].any()) for start, end in predicted_events]

    precision = (
        float(np.mean(predicted_hits)) if predicted_events else 0.0
    )
    recall = float(np.mean(true_hits)) if true_events else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0 else 0.0
    )

    delays = []
    normalized_delays = []
    for (start, end), hit in zip(true_events, true_hits):
        if not hit:
            continue
        first = int(np.flatnonzero(predicted[start:end])[0])
        delays.append(first)
        normalized_delays.append(first / float(end - start))

    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "true_events": int(len(true_events)),
        "predicted_events": int(len(predicted_events)),
        "detected_events": int(sum(true_hits)),
        "correct_predicted_events": int(sum(predicted_hits)),
        "mean_detection_delay": float(np.mean(delays)) if delays else None,
        "mean_normalized_detection_delay": (
            float(np.mean(normalized_delays)) if normalized_delays else None
        ),
    }
