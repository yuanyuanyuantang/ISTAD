#!/usr/bin/env python3
"""Verify completeness, ranges, and frozen-source integrity for Stage C v2."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


PROTOCOL_ID = "istad_interpretability_stage_c_v2_permutation_invariant"
SEEDS = (87, 90, 98)
PAIR_KEYS = ("87-90", "87-98", "90-98")
DATASET_ORDER = ("EXA", "PSM", "SMD", "SWAT")
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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_scalar(value, low, high, label, allow_none=False):
    if value is None and allow_none:
        return
    require(isinstance(value, (int, float)), f"{label}: not numeric")
    require(math.isfinite(float(value)), f"{label}: not finite")
    require(low - 1e-12 <= float(value) <= high + 1e-12, f"{label}: out of range")


def check_summary(summary, count, low, high, label):
    require(summary["count"] == count, f"{label}: count mismatch")
    for field in ("mean", "median", "p05", "p95", "min", "max"):
        check_scalar(summary[field], low, high, f"{label}.{field}")
    require(summary["min"] <= summary["p05"] + 1e-12, f"{label}: min/p05 order")
    require(summary["p05"] <= summary["median"] + 1e-12, f"{label}: p05/median order")
    require(summary["median"] <= summary["p95"] + 1e-12, f"{label}: median/p95 order")
    require(summary["p95"] <= summary["max"] + 1e-12, f"{label}: p95/max order")


def verify_dataset(name, block, repo_root):
    n_points = int(block["n_event_points"])
    n_variables = int(block["n_variables"])
    n_edges = int(block["n_hyperedges"])
    require(n_points > 0 and n_variables > 1 and n_edges > 0, f"{name}: invalid dimensions")

    evidence = block["one_step_residual_evidence"]
    check_scalar(evidence["repeat_max_abs_diff"], 0.0, 1e-12, f"{name}.evidence")
    require(evidence["seed_input_absent_by_construction"] is True, f"{name}: seed flag")
    require(len(evidence["event_array_sha256_float64_c_order"]) == 64, f"{name}: evidence hash")

    require(set(block["seed_runs"]) == {str(seed) for seed in SEEDS}, f"{name}: seed set")
    for seed in SEEDS:
        seed_block = block["seed_runs"][str(seed)]
        checkpoint = repo_root / seed_block["checkpoint"]
        require(checkpoint.is_file(), f"{name} seed {seed}: checkpoint missing")
        require(
            sha256_file(checkpoint) == seed_block["checkpoint_sha256"],
            f"{name} seed {seed}: checkpoint hash mismatch",
        )
        require(
            seed_block["event_incidence_shape"] == [n_points, n_variables, n_edges],
            f"{name} seed {seed}: incidence shape mismatch",
        )
        require(len(seed_block["event_incidence_npy_sha256"]) == 64, f"{name}: incidence hash")
        require(len(seed_block["peak_edge_evidence"]) == n_edges, f"{name}: d length")
        for index, value in enumerate(seed_block["peak_edge_evidence"]):
            check_scalar(value, 0.0, float("inf"), f"{name}.seed{seed}.d{index}")

    require(set(block["pairs"]) == set(PAIR_KEYS), f"{name}: pair set")
    for pair_key in PAIR_KEYS:
        pair = block["pairs"][pair_key]
        invariant = pair["permutation_invariant"]
        aligned = pair["hungarian_aligned_auxiliary"]
        neighbor_k = int(invariant["neighbor_k"])
        require(1 <= neighbor_k < n_variables, f"{name} {pair_key}: neighbor k")
        check_summary(invariant["top1_partition_ari"], n_points, -1.0, 1.0,
                      f"{name}.{pair_key}.ARI")
        check_summary(invariant["top1_partition_nmi"], n_points, 0.0, 1.0,
                      f"{name}.{pair_key}.NMI")
        check_summary(invariant["co_membership_cosine"], n_points, 0.0, 1.0,
                      f"{name}.{pair_key}.co_membership_cosine")
        check_summary(invariant["co_membership_relative_frobenius"], n_points,
                      0.0, float("inf"), f"{name}.{pair_key}.relative_frobenius")
        check_summary(invariant[f"top{neighbor_k}_neighbor_jaccard"],
                      n_points * n_variables, 0.0, 1.0,
                      f"{name}.{pair_key}.neighbor_jaccard")

        matching = aligned["a_to_b_hyperedge_match"]
        require(sorted(matching) == list(range(n_edges)), f"{name} {pair_key}: matching")
        check_summary(aligned["matched_incidence_column_cosine"], n_edges, 0.0, 1.0,
                      f"{name}.{pair_key}.matched_incidence")
        check_summary(aligned["aligned_top1_assignment_agreement"], n_points, 0.0, 1.0,
                      f"{name}.{pair_key}.aligned_agreement")
        check_summary(aligned["aligned_top1_assignment_jaccard"], n_points, 0.0, 1.0,
                      f"{name}.{pair_key}.aligned_jaccard")
        check_scalar(aligned["aligned_peak_edge_evidence_cosine"], -1.0, 1.0,
                     f"{name}.{pair_key}.d_cosine")
        check_scalar(aligned["aligned_peak_edge_evidence_kendall_tau_b"], -1.0, 1.0,
                     f"{name}.{pair_key}.d_tau", allow_none=True)


def atomic_write(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    result_dir = Path(args.result_dir).resolve()
    result_path = result_dir / "stability_v2.json"
    require(result_path.is_file(), f"result file missing: {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    expected_datasets = args.datasets or list(DATASET_ORDER)

    require(payload["protocol"] == PROTOCOL_ID, "protocol mismatch")
    require(payload["status"] == "COMPLETED", "run status is not COMPLETED")
    require(payload["seeds"] == list(SEEDS), "seed list mismatch")
    require(payload["datasets_requested"] == expected_datasets, "requested dataset order mismatch")
    require(set(payload["datasets"]) == set(expected_datasets), "dataset result set mismatch")

    source_checks = {}
    for relative_path, expected in EXPECTED_SOURCE_HASHES.items():
        actual = sha256_file(repo_root / relative_path)
        require(actual == expected, f"frozen source changed: {relative_path}")
        source_checks[relative_path] = "PASS"
    require(
        payload["old_stability_preserved"]["sha256"]
        == EXPECTED_SOURCE_HASHES["analysis/v4hg_interpretability/stability.json"],
        "payload does not preserve the v1 stability hash",
    )

    for name in expected_datasets:
        verify_dataset(name, payload["datasets"][name], repo_root)

    verification = {
        "status": "PASS",
        "protocol": PROTOCOL_ID,
        "datasets": expected_datasets,
        "dataset_count": len(expected_datasets),
        "checkpoint_count": 3 * len(expected_datasets),
        "pair_count": 3 * len(expected_datasets),
        "frozen_source_checks": source_checks,
        "old_stability_preserved": True,
        "result_sha256": sha256_file(result_path),
    }
    atomic_write(
        result_dir / "VERIFICATION.json",
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
