#!/usr/bin/env python3
"""CPU end-to-end smoke run for the V7 prior/model/calibration artifact path."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from models.ISTAD import Model
from utils.causal_prior import EntitywiseComponentECDF, fit_signed_causal_prior


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "analysis" / "v7_causal_hypergraph" / "smoke_artifact.npz"


def _series(seed, anomaly=False):
    rng = np.random.default_rng(seed)
    parts = []
    labels = []
    for entity in range(2):
        context = rng.normal(size=(64, 2))
        target = np.zeros(64)
        for time in range(1, 64):
            target[time] = (
                0.75 * target[time - 1]
                + 0.45 * context[time, 0]
                - 0.25 * context[time, 1]
                + rng.normal(scale=0.05)
            )
        entity_context = np.zeros((64, 2))
        entity_context[:, entity] = 1.0
        label = np.zeros(64, dtype=np.int8)
        if anomaly:
            target[40:44] += 4.0 * (entity + 1)
            label[40:44] = 1
        parts.append(np.column_stack([target, context, entity_context]))
        labels.append(label)
    return np.concatenate(parts), np.concatenate(labels), np.repeat([0, 1], 64)


def _windows(points, labels, entity_ids, width=16):
    point_windows, label_windows, id_windows = [], [], []
    for entity in np.unique(entity_ids):
        subset = points[entity_ids == entity]
        subset_labels = labels[entity_ids == entity]
        for start in range(0, len(subset) - width + 1, width):
            point_windows.append(subset[start:start + width])
            label_windows.append(subset_labels[start:start + width])
            id_windows.append(np.full(width, entity))
    return (
        np.asarray(point_windows, dtype=np.float32),
        np.asarray(label_windows, dtype=np.int8),
        np.asarray(id_windows, dtype=np.int16),
    )


def main():
    torch.manual_seed(7)
    train_points, train_labels, train_entities = _series(7, anomaly=False)
    test_points, test_labels, test_entities = _series(11, anomaly=True)
    prior, signs, _, prior_metadata = fit_signed_causal_prior(
        train_points, [0], lag=1, ridge=1e-2, topk=3,
        entity_ids=train_entities,
    )

    args = SimpleNamespace(
        task_name="anomaly_detection", seq_len=16, enc_in=5, c_out=1,
        features="M", istad_arch="v7", istad_objective="target_forecast",
        istad_target_features="0", istad_forecast_lag=1,
        istad_dual=0, istad_revin=0, istad_evidence_head=0, istad_denoise=0,
        istad_v7_relation_dim=8, istad_v7_prior_topk=3,
        istad_v7_prior_strength=1.0, istad_v7_prior_floor=0.01,
        istad_v7_prior_lambda=0.05, istad_v7_relation_score_weight=0.25,
        istad_recon_hid_dim=16, istad_dropout=0.0,
    )
    model = Model(args)
    model.set_causal_prior(prior, signs)
    exp = object.__new__(Exp_Anomaly_Detection)
    exp.args = args
    exp.model = model

    train_x, _, train_ids = _windows(
        train_points, train_labels, train_entities
    )
    test_x, test_y, test_ids = _windows(
        test_points, test_labels, test_entities
    )
    train_tensor = torch.from_numpy(train_x)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(2):
        optimizer.zero_grad()
        loss, _, _ = exp._v7_loss(train_tensor, nn.MSELoss())
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_output, train_aux = exp._forward_model(train_tensor, return_aux=True)
        test_output, test_aux = exp._forward_model(
            torch.from_numpy(test_x), return_aux=True
        )
        train_components = exp._v7_components(
            train_tensor, train_output, train_aux
        ).numpy().reshape(-1, 2)
        test_components = exp._v7_components(
            torch.from_numpy(test_x), test_output, test_aux
        ).numpy().reshape(-1, 2)

    train_entity = train_ids.reshape(-1)
    test_entity = test_ids.reshape(-1)
    calibrator = EntitywiseComponentECDF().fit(
        train_components, train_entity
    )
    train_calibrated = calibrator.transform(train_components, train_entity)
    test_calibrated = calibrator.transform(test_components, test_entity)
    weight = args.istad_v7_relation_score_weight
    score = (1.0 - weight) * test_calibrated[:, 0] + weight * test_calibrated[:, 1]
    metadata = {
        "synthetic_smoke_only": True,
        "prior": prior_metadata,
        "calibration": calibrator.metadata_dict(),
        "loss": float(loss.detach()),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        score=score.astype(np.float32),
        label=test_y.reshape(-1).astype(np.int8),
        entity_id=test_entity.astype(np.int16),
        v7_components=test_components.astype(np.float32),
        train_v7_components=train_components.astype(np.float32),
        v7_calibrated_components=test_calibrated.astype(np.float32),
        train_v7_calibrated_components=train_calibrated.astype(np.float32),
        v7_metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    if not np.isfinite(score).all():
        raise RuntimeError("non-finite V7 smoke score")
    print(json.dumps({
        "status": "passed",
        "artifact": str(OUTPUT.relative_to(REPO_ROOT)),
        "points": int(len(score)),
        "fields": sorted(np.load(OUTPUT).files),
    }, indent=2))


if __name__ == "__main__":
    main()
