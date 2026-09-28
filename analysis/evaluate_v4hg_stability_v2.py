#!/usr/bin/env python3
"""Permutation-invariant Stage C stability audit for frozen ISTAD models.

This is a methodological correction to Stage C v1. It preserves the frozen
datasets, events, seeds, checkpoints, and label-use boundary, but it does not
compare latent hyperedge IDs before accounting for label permutation.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_v4hg_interpretability as v1  # noqa: E402
from stage_c_v2_metrics import (  # noqa: E402
    aligned_hyperedge_metrics,
    relation_metrics,
)


EXPECTED_SOURCE_HASHES = {
    "analysis/evaluate_v4hg_interpretability.py":
        "7b2311f80d7a25ccb7299a173abcf1031d2a785c7cc18db7162ba8dd07a72dba",
    "analysis/v4hg_interpretability/PREREGISTERED_PROTOCOL.md":
        "a248c8f673bf86115938539c910593057ca251cfdd71668b806ff7e21d483450",
    "analysis/v4hg_interpretability/stability.json":
        "143fcc29b816c63ce4f87e338cb8058a7603fc40077ff704381cc2ce6361a68c",
    "code/ISTAD/models/ISTAD.py":
        "0504cd4cb9f82eca1738b4564060e08209847366fc3e32229be70fbbc5535fef",
    "code/ISTAD/models/istad_layers/hypergraph_attention.py":
        "f931c1dadb14c97e98b7ebf0cde866d0d601308a590e57210f2df13430e79d0b",
    "code/ISTAD/utils/innovation.py":
        "fc13b6fe5a96a0dc7a04ae61a46f3fb9098a7ea84178aeb057a09cc19b47674e",
    "code/ISTAD/data_provider/data_factory.py":
        "d163ef87f9bfe8d6606ddfb908d415f588578c05d8f96beb1c275362607f7914",
    "code/ISTAD/data_provider/data_loader.py":
        "4facaac1eb3d20cea100b6cae0e56ee2ae04b1e58e6b448cbbedfd619d4e6704",
}

PROTOCOL_ID = "istad_interpretability_stage_c_v2_permutation_invariant"
PAIR_KEYS = ("87-90", "87-98", "90-98")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array):
    canonical = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(canonical.view(np.uint8)).hexdigest()


def verify_frozen_sources():
    statuses = {}
    for relative_path, expected in EXPECTED_SOURCE_HASHES.items():
        path = v1.REPO_ROOT / relative_path
        if not path.is_file():
            raise RuntimeError(f"required frozen source is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen source mismatch for {relative_path}: {actual} != {expected}"
            )
        statuses[relative_path] = actual
    return statuses


def atomic_write(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def assert_frozen_event(name, geometry, old_stability):
    if name not in old_stability.get("datasets", {}):
        raise RuntimeError(f"v1 stability file has no frozen event for {name}")
    expected = old_stability["datasets"][name]
    actual_event = geometry["event"]
    expected_event = expected["event"]
    for key in ("start", "end", "entity"):
        if actual_event[key] != expected_event[key]:
            raise RuntimeError(
                f"{name}: frozen event {key} changed: "
                f"{actual_event[key]} != {expected_event[key]}"
            )
    if actual_event["length"] != expected["n_event_points"]:
        raise RuntimeError(f"{name}: frozen event length changed")


def evidence_identity_record(ctx, geometry):
    event = geometry["event"]
    repeated = ctx["scorer"].feature_evidence(
        ctx["test_points"], ctx["test_entities"]
    )[event["start"]:event["end"] + 1]
    reference = np.asarray(geometry["evidence_event"], dtype=np.float64)
    repeated = np.asarray(repeated, dtype=np.float64)
    maximum_difference = float(np.max(np.abs(reference - repeated)))
    if maximum_difference > 1e-12:
        raise RuntimeError(
            f"{ctx['name']}: repeated residual evidence differs by "
            f"{maximum_difference:.3e}"
        )
    return {
        "definition": (
            "training-calibrated one-step residual evidence; its computation "
            "contains no model seed or checkpoint input"
        ),
        "event_array_sha256_float64_c_order": sha256_array(reference),
        "repeat_max_abs_diff": maximum_difference,
        "seed_input_absent_by_construction": True,
    }


def export_seed_incidence(name, ctx, geometry, seed, device, destination):
    model, checkpoint_path = v1._load_model(name, ctx, seed, device)
    incidence = v1._forward_incidence(
        model, geometry["selected_windows"], device
    )
    event_low = geometry["event"]["start"] - geometry["window_low"] * ctx["W"]
    event_high = geometry["event"]["end"] - geometry["window_low"] * ctx["W"]
    event_incidence = np.ascontiguousarray(
        incidence[event_low:event_high + 1], dtype=np.float32
    )
    if event_incidence.shape[0] != geometry["event"]["length"]:
        raise RuntimeError(f"{name} seed {seed}: incidence/event length mismatch")
    if event_incidence.shape[1] != ctx["C"]:
        raise RuntimeError(f"{name} seed {seed}: incidence/variable mismatch")
    if not np.all(np.isfinite(event_incidence)) or np.any(event_incidence < 0):
        raise RuntimeError(f"{name} seed {seed}: invalid incidence values")

    peak = geometry["peak_block_local"]
    peak_edge_evidence = v1._edge_evidence(
        geometry["evidence_rows"][peak:peak + 1],
        incidence[peak:peak + 1],
    )[0]
    np.save(destination, event_incidence, allow_pickle=False)
    incidence_file_hash = sha256_file(destination)
    record = {
        "checkpoint": str(checkpoint_path.relative_to(v1.REPO_ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "event_incidence_npy_sha256": incidence_file_hash,
        "event_incidence_shape": [int(value) for value in event_incidence.shape],
        "peak_edge_evidence": [float(value) for value in peak_edge_evidence],
    }

    del event_incidence
    del incidence
    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return record


def run_dataset(name, data_root, device, scratch, old_stability, neighbor_k, chunk_size):
    print(f"=== preparing frozen Stage C v2 inputs: {name} ===", flush=True)
    ctx = v1.prepare_dataset(name, data_root)
    geometry = v1._case_geometry(ctx)
    assert_frozen_event(name, geometry, old_stability)
    evidence_record = evidence_identity_record(ctx, geometry)

    seed_records = {}
    incidence_paths = {}
    for seed in v1.SEEDS:
        print(f"{name}: forwarding seed {seed}", flush=True)
        path = Path(scratch) / f"{name}_seed{seed}_event_incidence.npy"
        seed_records[str(seed)] = export_seed_incidence(
            name, ctx, geometry, seed, device, path
        )
        incidence_paths[seed] = path

    pairs = {}
    for i, seed_a in enumerate(v1.SEEDS):
        for seed_b in v1.SEEDS[i + 1:]:
            pair_key = f"{seed_a}-{seed_b}"
            print(f"{name}: permutation-invariant comparison {pair_key}", flush=True)
            incidence_a = np.load(incidence_paths[seed_a], mmap_mode="r")
            incidence_b = np.load(incidence_paths[seed_b], mmap_mode="r")
            invariant = relation_metrics(
                incidence_a,
                incidence_b,
                neighbor_k=neighbor_k,
                chunk_size=chunk_size,
            )
            aligned = aligned_hyperedge_metrics(
                incidence_a,
                incidence_b,
                seed_records[str(seed_a)]["peak_edge_evidence"],
                seed_records[str(seed_b)]["peak_edge_evidence"],
                chunk_size=chunk_size,
            )
            pairs[pair_key] = {
                "permutation_invariant": invariant,
                "hungarian_aligned_auxiliary": aligned,
            }
            del incidence_a
            del incidence_b

    return {
        "event": {
            "start": int(geometry["event"]["start"]),
            "end": int(geometry["event"]["end"]),
            "entity": geometry["event"]["entity"],
            "selection": "frozen v1 longest-event rule; no reselection",
        },
        "n_event_points": int(geometry["event"]["length"]),
        "n_variables": int(ctx["C"]),
        "n_hyperedges": int(seed_records[str(v1.SEEDS[0])]["event_incidence_shape"][2]),
        "one_step_residual_evidence": evidence_record,
        "seed_runs": seed_records,
        "pairs": pairs,
    }


def metric_mean(pair, section, metric):
    return pair[section][metric]["mean"]


def write_results_markdown(payload):
    lines = [
        "# ISTAD Stage C v2: permutation-invariant stability",
        "",
        "This methodological correction keeps the frozen events, seeds, checkpoints, "
        "and label-use boundary. Latent hyperedge IDs are not compared before "
        "permutation alignment. All values are descriptive; no p-values or "
        "significance claims are produced.",
        "",
        "## Seed-independent variable evidence",
        "",
        "| Dataset | Event evidence repeat max abs diff | Event evidence SHA256 |",
        "|---|---:|---|",
    ]
    for name in v1.DATASET_ORDER:
        if name not in payload["datasets"]:
            continue
        evidence = payload["datasets"][name]["one_step_residual_evidence"]
        lines.append(
            f"| {name} | {evidence['repeat_max_abs_diff']:.3e} | "
            f"`{evidence['event_array_sha256_float64_c_order']}` |"
        )

    lines.extend([
        "",
        "## Permutation-invariant and aligned relation stability",
        "",
        "| Dataset | Seed pair | ARI | NMI | Co-membership cosine | "
        "Top-k neighbor Jaccard | Matched incidence cosine | "
        "Aligned assignment agreement | Aligned peak-d Kendall tau-b |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in v1.DATASET_ORDER:
        if name not in payload["datasets"]:
            continue
        for pair_key in PAIR_KEYS:
            pair = payload["datasets"][name]["pairs"][pair_key]
            invariant = pair["permutation_invariant"]
            aligned = pair["hungarian_aligned_auxiliary"]
            neighbor_metric = f"top{invariant['neighbor_k']}_neighbor_jaccard"
            tau = aligned["aligned_peak_edge_evidence_kendall_tau_b"]
            tau_text = "n/a" if tau is None else f"{tau:.3f}"
            lines.append(
                f"| {name} | {pair_key} | "
                f"{metric_mean(pair, 'permutation_invariant', 'top1_partition_ari'):.3f} | "
                f"{metric_mean(pair, 'permutation_invariant', 'top1_partition_nmi'):.3f} | "
                f"{metric_mean(pair, 'permutation_invariant', 'co_membership_cosine'):.3f} | "
                f"{metric_mean(pair, 'permutation_invariant', neighbor_metric):.3f} | "
                f"{metric_mean(pair, 'hungarian_aligned_auxiliary', 'matched_incidence_column_cosine'):.3f} | "
                f"{metric_mean(pair, 'hungarian_aligned_auxiliary', 'aligned_top1_assignment_agreement'):.3f} | "
                f"{tau_text} |"
            )

    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The one-step residual evidence is the primary variable-level explanation. "
        "The HGAT incidence is an auxiliary relational view, and its stability is "
        "assessed through induced variable relations and explicitly aligned latent "
        "hyperedges. These results do not test physical root-cause correctness or "
        "the detection benefit of dynamic topology.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/data/modeluse/TS/ISTAD/dataset")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=sorted(v1.DATASETS),
        help="EXA-only is permitted for smoke testing; the frozen full run uses all four",
    )
    parser.add_argument(
        "--output-dir",
        default=str(v1.OUT_DIR / "stage_c_v2"),
    )
    parser.add_argument("--neighbor-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--forward-batch", type=int, default=256)
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="retain event incidence arrays; disabled in the normal delivery run",
    )
    args = parser.parse_args()

    if args.neighbor_k < 1 or args.chunk_size < 1 or args.forward_batch < 1:
        parser.error("neighbor-k, chunk-size, and forward-batch must be positive")

    source_hashes = verify_frozen_sources()
    old_stability_path = v1.OUT_DIR / "stability.json"
    old_stability = json.loads(old_stability_path.read_text(encoding="utf-8"))
    selected = args.datasets or v1.DATASET_ORDER
    output_dir = Path(args.output_dir).resolve()
    result_path = output_dir / "stability_v2.json"
    if result_path.exists():
        raise RuntimeError(
            f"refusing to overwrite an existing Stage C v2 result: {result_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = v1._resolve_device(args.device)
    v1.FORWARD_BATCH = int(args.forward_batch)
    scratch_base = output_dir / "_scratch"
    scratch_base.mkdir(exist_ok=False)
    temporary = None
    try:
        if args.keep_raw:
            scratch = output_dir / "raw_incidence"
            scratch.mkdir(exist_ok=False)
        else:
            temporary = tempfile.TemporaryDirectory(prefix="incidence_", dir=scratch_base)
            scratch = Path(temporary.name)

        datasets = {}
        for name in selected:
            datasets[name] = run_dataset(
                name,
                args.data_root,
                device,
                scratch,
                old_stability,
                args.neighbor_k,
                args.chunk_size,
            )

        payload = {
            "protocol": PROTOCOL_ID,
            "status": "COMPLETED",
            "revision_reason": (
                "Stage C v1 compared non-identifiable latent hyperedge IDs directly; "
                "v2 uses permutation-invariant induced relations and event-wide "
                "Hungarian alignment"
            ),
            "analysis_scope": "descriptive methodological correction",
            "label_use": (
                "frozen test labels locate the previously selected demonstration "
                "events only; no threshold, score, model, or metric is selected"
            ),
            "datasets_requested": list(selected),
            "seeds": [int(seed) for seed in v1.SEEDS],
            "neighbor_k_requested": int(args.neighbor_k),
            "source_sha256": source_hashes,
            "old_stability_preserved": {
                "path": str(old_stability_path.relative_to(v1.REPO_ROOT)),
                "sha256": sha256_file(old_stability_path),
            },
            "definitions": {
                "top1_partition_ari_nmi": (
                    "ARI and arithmetic-mean NMI between per-variable top-1 "
                    "hyperedge partitions at each event point; invariant to labels"
                ),
                "co_membership": (
                    "row-normalize H over hyperedges, form P=H H^T, exclude the "
                    "diagonal, and compare upper triangles at each event point"
                ),
                "neighbor_jaccard": (
                    "Jaccard overlap of each variable's top-k neighbors induced by P"
                ),
                "hungarian_alignment": (
                    "one event-wide maximum-cosine matching between flattened "
                    "hyperedge incidence columns for each seed pair"
                ),
                "aligned_peak_edge_evidence": (
                    "cosine and Kendall tau-b between d vectors at the shared, "
                    "seed-independent residual-evidence peak after matching"
                ),
            },
            "datasets": datasets,
        }
        atomic_write(result_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        atomic_write(
            output_dir / "STABILITY_V2_RESULTS.md",
            write_results_markdown(payload),
        )
        print(f"Stage C v2 result: {result_path}", flush=True)
    finally:
        if temporary is not None:
            temporary.cleanup()
        if scratch_base.exists() and not any(scratch_base.iterdir()):
            scratch_base.rmdir()
        elif scratch_base.exists() and not args.keep_raw:
            shutil.rmtree(scratch_base)


if __name__ == "__main__":
    main()
