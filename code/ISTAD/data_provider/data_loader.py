import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

HUGGINGFACE_REPO = "thuml/Time-Series-Library"

# Per-entity lengths in the lexicographic filename order used by the flattened
# release (group 3 is 3-1, 3-10, 3-11, 3-2, ..., 3-9).  The totals
# (708405/708420) exactly match the TSLib arrays.  Source statistics: TimeSeAD,
# TMLR 2023, Appendix B, Table 2.
SMD_TRAIN_LENGTHS = (
    28479, 23694, 23702, 23706, 23705, 23688, 23697, 23698,
    23693, 23699, 23688, 23689, 23688, 28743, 23696, 23702,
    28722, 28700, 23692, 28695, 23702, 23703, 23687, 23690,
    28726, 28705, 28703, 28713,
)
SMD_TEST_LENGTHS = (
    28479, 23694, 23703, 23707, 23706, 23689, 23697, 23699,
    23694, 23700, 23689, 23689, 23689, 28743, 23696, 23703,
    28722, 28700, 23693, 28696, 23703, 23703, 23687, 23691,
    28726, 28705, 28704, 28713,
)

# NASA Telemanom entities in the exact order used by the flattened TSLib arrays.
# The test boundaries were verified by regenerating both official label arrays
# exactly from labeled_anomalies.csv.  Keeping these boundaries prevents windows,
# causal lag pairs, and point adjustment from crossing unrelated spacecraft
# channels.  Totals: MSL 58,317/73,729; SMAP 135,183/427,617.
MSL_TRAIN_LENGTHS = (
    2158, 764, 3675, 2074, 1451, 2244, 2598, 2511, 3342,
    2209, 2208, 2037, 2076, 2032, 1565, 1587, 4308, 3969,
    2880, 3682, 926, 1145, 1145, 2272, 2272, 748, 439,
)
MSL_TEST_LENGTHS = (
    2264, 2051, 2625, 2158, 2191, 3422, 3922, 5054, 2487,
    2277, 2277, 2127, 2038, 2303, 2049, 2156, 6100, 3535,
    6100, 2856, 1827, 2430, 2430, 2217, 2218, 1519, 1096,
)
SMAP_TRAIN_LENGTHS = (
    2880, 2648, 2736, 2690, 705, 682, 2879, 762, 762, 2435,
    2849, 2611, 312, 1490, 2880, 2880, 2833, 2561, 2594,
    2583, 2602, 2583, 2880, 2880, 2880, 2880, 2880, 2880,
    2880, 2880, 2880, 2880, 2769, 2880, 2880, 2869, 2861,
    2880, 2820, 2478, 2624, 2551, 2881, 2446, 2872, 2855,
    2609, 2853, 2874, 2818, 2875, 2855, 2876,
)
SMAP_TEST_LENGTHS = (
    8640, 7914, 8205, 8080, 4693, 4453, 8631, 8375, 8434,
    8044, 8509, 7431, 7918, 7663, 8595, 8640, 8473, 7628,
    7884, 7642, 7874, 7406, 8516, 8505, 8514, 8512, 8640,
    8532, 8307, 8354, 8294, 8300, 8310, 8532, 8302, 8584,
    8626, 8376, 8469, 7361, 7907, 7632, 8640, 8029, 8505,
    8493, 7783, 8071, 7244, 7331, 8612, 8625, 8579,
)


def _train_val_split(args, train):
    """Return the legacy overlapping split or a strict chronological holdout.

    The legacy loaders trained on the complete array and reused its final 20% as
    validation.  ISTAD-v3 opts into a disjoint split so early stopping is based
    on unseen normal windows.  The default keeps old checkpoints reproducible.
    """
    split = int(len(train) * 0.8)
    if bool(int(getattr(args, "istad_holdout_val", 0) or 0)):
        return train[:split], train[split:]
    return train, train[split:]


def _scale_train_val(args, scaler, train):
    """Fit normalization without leaking strict validation statistics."""
    if bool(int(getattr(args, "istad_holdout_val", 0) or 0)):
        train_raw, val_raw = _train_val_split(args, train)
        scaler.fit(train_raw)
        return scaler.transform(train_raw), scaler.transform(val_raw)
    scaler.fit(train)
    scaled = scaler.transform(train)
    return _train_val_split(args, scaled)


def _parts_from_lengths(values, lengths):
    """Split a flattened multi-entity array and validate its provenance."""
    lengths = tuple(int(length) for length in lengths)
    if sum(lengths) != len(values):
        raise ValueError(
            f"Entity lengths sum to {sum(lengths)}, but array has {len(values)} rows"
        )
    offsets = np.cumsum((0,) + lengths)
    return [values[offsets[i]:offsets[i + 1]] for i in range(len(lengths))]


def _window_starts(lengths, win_size, step):
    """Return flattened starts without ever crossing an entity boundary."""
    starts = []
    offset = 0
    for length in lengths:
        if length >= win_size:
            starts.extend(offset + start for start in range(0, length - win_size + 1, step))
        offset += length
    return np.asarray(starts, dtype=np.int64)


def _scale_segmented_train_val(scaler, train, lengths, ratio=0.8):
    """Chronologically hold out the tail of every entity and fit on heads only."""
    parts = _parts_from_lengths(train, lengths)
    split_points = [int(len(part) * ratio) for part in parts]
    train_parts = [part[:split] for part, split in zip(parts, split_points)]
    val_parts = [part[split:] for part, split in zip(parts, split_points)]
    scaler.fit(np.concatenate(train_parts, axis=0))
    scaled_train = [scaler.transform(part) for part in train_parts]
    scaled_val = [scaler.transform(part) for part in val_parts]
    return (
        np.concatenate(scaled_train, axis=0),
        np.concatenate(scaled_val, axis=0),
        tuple(len(part) for part in scaled_train),
        tuple(len(part) for part in scaled_val),
    )


def _append_entity_context(values, lengths):
    """Append a constant one-hot entity identity to every point in each part."""
    values = np.asarray(values)
    lengths = tuple(int(length) for length in lengths)
    if sum(lengths) != len(values):
        raise ValueError(
            f"Entity lengths sum to {sum(lengths)}, but values have {len(values)} rows"
        )
    entity_ids = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    context = np.eye(len(lengths), dtype=values.dtype)[entity_ids]
    return np.concatenate([values, context], axis=1)


def _configure_segmented_loader(
    dataset,
    args,
    dataset_name,
    train_data,
    test_data,
    train_lengths,
    test_lengths,
):
    """Scale/split a flattened multi-entity dataset without crossing entities."""
    dataset.entity_aware = bool(int(getattr(args, "istad_entity_aware", 0) or 0))
    if not dataset.entity_aware:
        dataset.train, dataset.val = _scale_train_val(args, dataset.scaler, train_data)
        dataset.test = dataset.scaler.transform(test_data)
        return

    if not bool(int(getattr(args, "istad_holdout_val", 0) or 0)):
        raise ValueError(
            f"{dataset_name} entity-aware mode requires --istad_holdout_val 1"
        )
    dataset.train, dataset.val, fit_lengths, val_lengths = (
        _scale_segmented_train_val(dataset.scaler, train_data, train_lengths)
    )
    _parts_from_lengths(test_data, test_lengths)
    dataset.test = dataset.scaler.transform(test_data)
    dataset.segment_lengths = {
        "train": fit_lengths,
        "val": val_lengths,
        "test": test_lengths,
    }
    dataset.raw_feature_count = int(train_data.shape[1])
    dataset.entity_context_dim = 0
    if bool(int(getattr(args, "istad_entity_context", 0) or 0)):
        dataset.train = _append_entity_context(dataset.train, fit_lengths)
        dataset.val = _append_entity_context(dataset.val, val_lengths)
        dataset.test = _append_entity_context(dataset.test, test_lengths)
        dataset.entity_context_dim = len(train_lengths)
        expected_width = int(getattr(args, "enc_in", dataset.train.shape[1]))
        if dataset.train.shape[1] != expected_width:
            raise ValueError(
                "--istad_entity_context 1 appends one node per entity: "
                f"expected --enc_in {dataset.train.shape[1]}, got {expected_width}"
            )
    active_lengths = dataset.segment_lengths[dataset.flag]
    dataset.window_starts = _window_starts(
        active_lengths, dataset.win_size, dataset.step
    )
    boundaries = np.cumsum(active_lengths)
    dataset.window_entity_ids = np.searchsorted(
        boundaries, dataset.window_starts, side="right"
    ).astype(np.int16)
    dataset.point_entity_ids = np.repeat(
        dataset.window_entity_ids, dataset.win_size
    )


def _entity_window_item(dataset, index):
    start = int(dataset.window_starts[index])
    if dataset.flag == "train":
        values = dataset.train
        labels = dataset.test_labels[0:dataset.win_size]
    elif dataset.flag == "val":
        values = dataset.val
        labels = dataset.test_labels[0:dataset.win_size]
    else:
        values = dataset.test
        labels = dataset.test_labels[start:start + dataset.win_size]
    return (
        np.float32(values[start:start + dataset.win_size]),
        np.float32(labels),
    )


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optional dependency 'datasets' is required for remote dataset download. "
            "Either install it (`pip install datasets`) or place dataset files under --root_path."
        ) from exc

    return load_dataset(*args, **kwargs)


def _hf_hub_download(*args, **kwargs):
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optional dependency 'huggingface_hub' is required for remote dataset download. "
            "Either install it (`pip install huggingface_hub`) or place dataset files under --root_path."
        ) from exc

    return hf_hub_download(*args, **kwargs)


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = os.path.join(root_path, "train.csv")
        test_path = os.path.join(root_path, "test.csv")
        label_path = os.path.join(root_path, "test_label.csv")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            test_label_df = pd.read_csv(label_path)
        else:
            ds_data = _load_dataset(HUGGINGFACE_REPO, name="PSM-data", cache_dir=root_path)
            ds_label = _load_dataset(HUGGINGFACE_REPO, name="PSM-label", cache_dir=root_path)
            train_df = ds_data["train"].to_pandas()
            test_df = ds_data["test"].to_pandas()
            test_label_df = ds_label[next(iter(ds_label))].to_pandas()

            os.makedirs(root_path, exist_ok=True)
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            test_label_df.to_csv(label_path, index=False)

        train_data = np.nan_to_num(train_df.values[:, 1:])
        self.train, self.val = _scale_train_val(args, self.scaler, train_data)

        test_data = np.nan_to_num(test_df.values[:, 1:])
        self.test = self.scaler.transform(test_data)

        self.test_labels = test_label_df.values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        if self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        if self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        if self.flag == "val":
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        if self.flag == "test":
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        return np.float32(
            self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        ), np.float32(
            self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        )


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = os.path.join(root_path, "MSL_train.npy")
        test_path = os.path.join(root_path, "MSL_test.npy")
        label_path = os.path.join(root_path, "MSL_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = _hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename="MSL/MSL_train.npy",
                repo_type="dataset",
                local_dir=root_path,
            )
            test_path = _hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename="MSL/MSL_test.npy",
                repo_type="dataset",
                local_dir=root_path,
            )
            label_path = _hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename="MSL/MSL_test_label.npy",
                repo_type="dataset",
                local_dir=root_path,
            )

            train_data = np.load(train_path)
            test_data = np.load(test_path)
            test_label = np.load(label_path)

        _configure_segmented_loader(
            self,
            args,
            "MSL",
            train_data,
            test_data,
            MSL_TRAIN_LENGTHS,
            MSL_TEST_LENGTHS,
        )
        self.test_labels = test_label

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        if self.entity_aware:
            print("MSL entity-aware windows:", len(self.window_starts))

    def __len__(self):
        if self.entity_aware:
            return len(self.window_starts)
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        if self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        if self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        if self.entity_aware:
            return _entity_window_item(self, index)
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        if self.flag == "val":
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        if self.flag == "test":
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        return np.float32(
            self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        ), np.float32(
            self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        )


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = os.path.join(root_path, "SMAP_train.npy")
        test_path = os.path.join(root_path, "SMAP_test.npy")
        label_path = os.path.join(root_path, "SMAP_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = _hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename="SMAP/SMAP_train.npy",
                repo_type="dataset",
                local_dir=root_path,
            )
            test_path = _hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename="SMAP/SMAP_test.npy",
                repo_type="dataset",
                local_dir=root_path,
            )
            label_path = _hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename="SMAP/SMAP_test_label.npy",
                repo_type="dataset",
                local_dir=root_path,
            )

            train_data = np.load(train_path)
            test_data = np.load(test_path)
            test_label = np.load(label_path)

        _configure_segmented_loader(
            self,
            args,
            "SMAP",
            train_data,
            test_data,
            SMAP_TRAIN_LENGTHS,
            SMAP_TEST_LENGTHS,
        )
        self.test_labels = test_label

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        if self.entity_aware:
            print("SMAP entity-aware windows:", len(self.window_starts))

    def __len__(self):
        if self.entity_aware:
            return len(self.window_starts)
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        if self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        if self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        if self.entity_aware:
            return _entity_window_item(self, index)
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        if self.flag == "val":
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        if self.flag == "test":
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        return np.float32(
            self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        ), np.float32(
            self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        )


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=95, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = os.path.join(root_path, "SMD_train.npy")
        test_path = os.path.join(root_path, "SMD_test.npy")
        label_path = os.path.join(root_path, "SMD_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = _hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename="SMD/SMD_train.npy",
                repo_type="dataset",
                local_dir=root_path,
            )
            test_path = _hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename="SMD/SMD_test.npy",
                repo_type="dataset",
                local_dir=root_path,
            )
            label_path = _hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename="SMD/SMD_test_label.npy",
                repo_type="dataset",
                local_dir=root_path,
            )

            train_data = np.load(train_path)
            test_data = np.load(test_path)
            test_label = np.load(label_path)

        _configure_segmented_loader(
            self,
            args,
            "SMD",
            train_data,
            test_data,
            SMD_TRAIN_LENGTHS,
            SMD_TEST_LENGTHS,
        )
        self.test_labels = test_label

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        if self.entity_aware:
            print("SMD entity-aware windows:", len(self.window_starts))

    def __len__(self):
        if self.entity_aware:
            return len(self.window_starts)
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        if self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        if self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        if self.entity_aware:
            start = int(self.window_starts[index])
            if self.flag == "train":
                return np.float32(self.train[start:start + self.win_size]), np.float32(
                    self.test_labels[0:self.win_size]
                )
            if self.flag == "val":
                return np.float32(self.val[start:start + self.win_size]), np.float32(
                    self.test_labels[0:self.win_size]
                )
            return np.float32(self.test[start:start + self.win_size]), np.float32(
                self.test_labels[start:start + self.win_size]
            )
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        if self.flag == "val":
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        if self.flag == "test":
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        return np.float32(
            self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        ), np.float32(
            self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        )


class EXATHLONSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=95, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = os.path.join(root_path, "Exathlon_train.npy")
        test_path = os.path.join(root_path, "Exathlon_test.npy")
        label_path = os.path.join(root_path, "Exathlon_test_label.npy")

        missing_paths = [
            path for path in [train_path, test_path, label_path]
            if not os.path.exists(path)
        ]
        if missing_paths:
            missing = ", ".join(missing_paths)
            raise FileNotFoundError(f"Missing EXATHLON dataset file(s): {missing}")

        train_data = np.load(train_path)
        test_data = np.load(test_path)
        test_label = np.load(label_path)

        self.train, self.val = _scale_train_val(args, self.scaler, train_data)
        self.test = self.scaler.transform(test_data)
        self.test_labels = test_label

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        if self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        if self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        if self.flag == "val":
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        if self.flag == "test":
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        return np.float32(
            self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        ), np.float32(
            self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        )


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=95, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = os.path.join(root_path, "swat_train2.csv")
        test_path = os.path.join(root_path, "swat2.csv")

        if all(os.path.exists(p) for p in [train_path, test_path]):
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
        else:
            ds = _load_dataset(HUGGINGFACE_REPO, name="SWaT", cache_dir=root_path)
            train_data = ds["train"].to_pandas()
            test_data = ds["test"].to_pandas()

            os.makedirs(root_path, exist_ok=True)
            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)

        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.train, self.val = _scale_train_val(args, self.scaler, train_data)
        self.test = self.scaler.transform(test_data)
        self.test_labels = labels

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        if self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        if self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        if self.flag == "val":
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        if self.flag == "test":
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]
            )
        return np.float32(
            self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        ), np.float32(
            self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        )
