#!/usr/bin/env python3
"""Reproducible model-only efficiency benchmark for frozen V4-HG checkpoints."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "ISTAD"))
from models.ISTAD import Model  # noqa: E402


DATASETS = {
    "EXA": {"long": "EXATHLON", "seq_len": 100, "channels": 19, "batch": 128,
            "kernel": 7, "edges": 9, "topk": 3},
    "PSM": {"long": "PSM", "seq_len": 64, "channels": 25, "batch": 128,
            "kernel": 7, "edges": 12, "topk": 5},
    "SMD": {"long": "SMD", "seq_len": 96, "channels": 38, "batch": 128,
            "kernel": 7, "edges": 19, "topk": 7},
    "SWAT": {"long": "SWAT", "seq_len": 96, "channels": 51, "batch": 64,
             "kernel": 15, "edges": 20, "topk": 10},
}


def _build(spec):
    return Model(SimpleNamespace(
        task_name="anomaly_detection", seq_len=spec["seq_len"],
        enc_in=spec["channels"], c_out=spec["channels"],
        istad_arch="legacy", istad_objective="reconstruct",
        istad_target_features="", istad_dual=0, istad_revin=0,
        istad_evidence_head=0, istad_branch_mode="hgat",
        istad_tcn_type="kan", istad_spatial_type="lite", istad_hgat_rank=8,
        istad_kernel_size=spec["kernel"], istad_n_hyperedges=spec["edges"],
        istad_k_top=spec["topk"], istad_recon_type="pointwise",
        istad_recon_hid_dim=32, istad_dropout=0.3,
    ))


def _checkpoint(dataset):
    spec = DATASETS[dataset]
    matches = list((REPO_ROOT / "checkpoints" / "ISTAD_v4_hgat_integrated").glob(
        f"anomaly_detection_{spec['long']}_*_v4hg_s87_0/checkpoint.pth"
    ))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one seed-87 checkpoint for {dataset}: {matches}")
    return matches[0]


def _timed(model, sample, warmup, repeats, device):
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample, None, None, None, return_aux=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(repeats):
            if device.type == "cuda":
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
                model(sample, None, None, None, return_aux=True)
                ender.record()
                torch.cuda.synchronize(device)
                times.append(float(starter.elapsed_time(ender)))
            else:
                start = time.perf_counter()
                model(sample, None, None, None, return_aux=True)
                times.append(1000.0 * (time.perf_counter() - start))
    peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return {
        "mean_ms_per_batch": float(statistics.mean(times)),
        "population_std_ms_per_batch": float(statistics.pstdev(times)),
        "median_ms_per_batch": float(statistics.median(times)),
        "mean_microseconds_per_window": float(1000.0 * statistics.mean(times) / len(sample)),
        "peak_allocated_bytes": peak,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "analysis" / "v4_hgat_paper" / "efficiency.json"
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    results = {}
    for dataset, spec in DATASETS.items():
        checkpoint = _checkpoint(dataset)
        model = _build(spec)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        total = sum(parameter.numel() for parameter in model.parameters())
        hgat = sum(
            parameter.numel() for parameter in model.backbone.feature_gat.parameters()
        )
        dataset_result = {
            "parameters": int(total),
            "hgat_parameters": int(hgat),
            "checkpoint_bytes": int(checkpoint.stat().st_size),
            "production_batch_size": int(spec["batch"]),
            "batch_1": _timed(
                model,
                torch.randn(1, spec["seq_len"], spec["channels"], device=device),
                args.warmup, args.repeats, device,
            ),
            "production_batch": _timed(
                model,
                torch.randn(
                    spec["batch"], spec["seq_len"], spec["channels"], device=device
                ),
                args.warmup, args.repeats, device,
            ),
        }
        train_times = []
        for seed in (87, 90, 98):
            result_matches = list((REPO_ROOT / "code" / "ISTAD" / "test_results").glob(
                f"anomaly_detection_{spec['long']}_*_v4hg_s{seed}_0/train_time_summary.json"
            ))
            if len(result_matches) == 1:
                train_times.append(float(json.loads(
                    result_matches[0].read_text()
                )["avg_epoch_train_time"]))
        dataset_result["observed_epoch_train_seconds"] = {
            "values": train_times,
            "mean": float(np.mean(train_times)),
            "population_std": float(np.std(train_times)),
        }
        results[dataset] = dataset_result
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "scope": "model_forward_with_dynamic_incidence_return; excludes_data_loading_and_VAR_fit",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    markdown = [
        "# V4-HG efficiency benchmark",
        "",
        f"> Device: {report['gpu_name'] or report['device']}; PyTorch {report['torch_version']}; "
        f"{args.warmup} warm-up + {args.repeats} measured passes.",
        "> Scope: model forward with dynamic incidence output; excludes data loading and VAR fitting.",
        "",
        "| Dataset | Parameters | HGAT parameters | Checkpoint | Batch | ms/batch | us/window | Peak allocated | Epoch train time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, values in results.items():
        production = values["production_batch"]
        training = values["observed_epoch_train_seconds"]
        markdown.append(
            f"| {dataset} | {values['parameters']:,} | {values['hgat_parameters']:,} | "
            f"{values['checkpoint_bytes'] / 1024.0:.1f} KiB | "
            f"{values['production_batch_size']} | {production['mean_ms_per_batch']:.3f}±"
            f"{production['population_std_ms_per_batch']:.3f} | "
            f"{production['mean_microseconds_per_window']:.2f} | "
            f"{production['peak_allocated_bytes'] / (1024.0 ** 2):.1f} MiB | "
            f"{training['mean']:.3f}±{training['population_std']:.3f} s |"
        )
    markdown.extend([
        "",
        "The epoch times are observational values from the three completed training runs;",
        "they are not a controlled cross-model speed comparison. PSM is much longer than",
        "the other training loaders, so epoch time must not be compared across datasets.",
        "",
    ])
    (args.output.parent / "EFFICIENCY.md").write_text("\n".join(markdown))
    print(args.output)


if __name__ == "__main__":
    main()
