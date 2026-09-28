#!/usr/bin/env python3
"""Recover the official TSLib MSL/SMAP NumPy files from local pickle copies.

The archived files contain plain NumPy arrays.  This utility validates their
types, canonical benchmark shapes, finiteness, label domain, and the SHA-256 of
the NumPy serialization published by thuml/Time-Series-Library before writing
anything.  Existing destination files are never silently replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT.parent / "上次的ISTAD" / "datasets" / "data" / "processed"
)
DEFAULT_DESTINATION = REPO_ROOT.parent / "ISTAD" / "dataset"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "analysis"
    / "independent_confirmation"
    / "dataset_conversion_manifest.json"
)

# SHA-256 values shown on the official Hugging Face file pages at commit
# ffdb671c1a34bbed9fa689dbc4bcbf1ac7c5ba4a.
FILES = {
    "MSL": {
        "train": {
            "shape": (58_317, 55),
            "sha256": "d767ecfc8ea344f7a6e78f815a84dc7923f600df1beb14359f3e73202148b058",
        },
        "test": {
            "shape": (73_729, 55),
            "sha256": "8b479ebeeabb63ff636e8adeb095a7bcc46a0cdc7befa189b45b05b1e90c9425",
        },
        "test_label": {
            "shape": (73_729,),
            "sha256": "23e120c30a645000f093ff5311de177601b8220ba3ca4281c6376883444f7b97",
        },
    },
    "SMAP": {
        "train": {
            "shape": (135_183, 25),
            "sha256": "1c823bed59f32d45a2e6323ac79bf7755c80c2ff68a2d3d7322e4dee40a11010",
        },
        "test": {
            "shape": (427_617, 25),
            "sha256": "458c3b354b6a602b711241c29fb50518dddb0c1c9248d7d2e1691a9c218d4ac0",
        },
        "test_label": {
            "shape": (427_617,),
            "sha256": "55f94fcccc3ae35216cfd6ac5e0e2016ad9d08d5ff0aca9f09916c6501380e0f",
        },
    },
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _serialize_npy(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def _load_validated_array(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    # The local archive is project-owned.  Its pickles were separately audited
    # to contain only NumPy ndarray reconstruction opcodes before this utility
    # was added; never point --source at untrusted pickle files.
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if type(value) is not np.ndarray:
        raise TypeError(f"{path}: expected exact numpy.ndarray, got {type(value)!r}")
    if value.shape != expected_shape:
        raise ValueError(f"{path}: expected shape {expected_shape}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{path}: contains NaN or infinite values")
    return value


def convert(source: Path, destination: Path, manifest_path: Path) -> dict:
    records = []
    prepared = []
    for dataset, parts in FILES.items():
        for part, specification in parts.items():
            stem = f"{dataset}_{part}"
            source_path = source / dataset / f"{stem}.pkl"
            array = _load_validated_array(source_path, specification["shape"])
            if part == "test_label" and not np.isin(array, (False, True, 0, 1)).all():
                raise ValueError(f"{source_path}: labels are not binary")
            payload = _serialize_npy(array)
            payload_sha256 = _sha256_bytes(payload)
            if payload_sha256 != specification["sha256"]:
                raise ValueError(
                    f"{source_path}: canonical .npy SHA-256 {payload_sha256} does not "
                    f"match official TSLib SHA-256 {specification['sha256']}"
                )
            destination_path = destination / dataset / f"{stem}.npy"
            if destination_path.exists():
                existing_sha256 = _sha256_file(destination_path)
                if existing_sha256 != payload_sha256:
                    raise FileExistsError(
                        f"refusing to overwrite {destination_path}: SHA-256 is "
                        f"{existing_sha256}, expected {payload_sha256}"
                    )
            prepared.append((destination_path, payload))
            records.append(
                {
                    "dataset": dataset,
                    "part": part,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "source_path": str(source_path),
                    "source_pickle_sha256": _sha256_file(source_path),
                    "destination_path": str(destination_path),
                    "destination_npy_sha256": payload_sha256,
                    "official_tslib_sha256_match": True,
                }
            )

    for destination_path, payload in prepared:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
        with temporary_path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(destination_path)

    manifest = {
        "status": "verified_exact_official_tslib_numpy_serializations",
        "official_repository": "thuml/Time-Series-Library",
        "official_commit": "ffdb671c1a34bbed9fa689dbc4bcbf1ac7c5ba4a",
        "conversion": "pickle.load followed by numpy.save(allow_pickle=False)",
        "records": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = convert(args.source, args.destination, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
