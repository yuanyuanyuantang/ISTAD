import argparse
import os

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, 'dataset_tranad')


DATASET_ALIASES = {
    'psm': 'PSM',
    'smd': 'SMD',
    'swat': 'SWaT',
    'exathlon': 'Exathlon',
}

DATASET_SPECS = {
    'PSM': {
        'folder': 'PSM',
        'sensors': 25,
        'scaling': 'none',
    },
    'SMD': {
        'folder': 'SMD',
        'sensors': 38,
        'scaling': 'none',
    },
    'SWaT': {
        'folder': 'SWAT',
        'sensors': 51,
        'scaling': 'auto',
    },
    'Exathlon': {
        'folder': 'EXATHLON',
        'sensors': 19,
        'scaling': 'auto',
    },
}


def resolve_data_root(data_root=None):
    return os.path.abspath(data_root) if data_root else DEFAULT_DATA_ROOT


def _resolve_child(parent, candidates, kind):
    for candidate in candidates:
        path = os.path.join(parent, candidate)
        if os.path.exists(path):
            return path

    if os.path.isdir(parent):
        entries = os.listdir(parent)
        lower_to_entry = {entry.lower(): entry for entry in entries}
        for candidate in candidates:
            entry = lower_to_entry.get(candidate.lower())
            if entry is not None:
                return os.path.join(parent, entry)

    searched = ', '.join(os.path.join(parent, candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing {kind}. Searched: {searched}")


def _resolve_folder(root, *candidates):
    return _resolve_child(root, candidates, 'dataset folder')


def _resolve_file(folder, *candidates):
    return _resolve_child(folder, candidates, 'dataset file')


def _try_resolve_file(folder, *candidates):
    try:
        return _resolve_file(folder, *candidates)
    except FileNotFoundError:
        return None


def canonical_dataset(dataset):
    key = dataset.strip().lower()
    if key not in DATASET_ALIASES:
        supported = ', '.join(DATASET_SPECS)
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported: {supported}")
    return DATASET_ALIASES[key]


def _binary_labels(labels):
    labels = np.asarray(labels)
    if labels.dtype.kind in 'OUS':
        flat_labels = labels.reshape(-1)
        normalized = pd.Series(flat_labels).astype(str).str.strip().str.lower()
        binary = (~normalized.isin(['0', 'normal', 'false', 'no'])).astype(np.float32).to_numpy()
        labels = binary.reshape(labels.shape)
    else:
        labels = (labels.astype(np.float32) > 0).astype(np.float32)

    if labels.ndim == 1:
        return labels.reshape(-1, 1)
    if labels.ndim == 2 and labels.shape[1] == 1:
        return labels.astype(np.float32)
    if labels.ndim >= 2:
        labels = np.any(labels.reshape(labels.shape[0], -1) > 0, axis=1).astype(np.float32)
        return labels.reshape(-1, 1)

    return labels.reshape(-1, 1)


def _nan_to_float32(data):
    return np.nan_to_num(np.asarray(data, dtype=np.float32))


def _train_minmax_scale(train, test):
    min_values = np.nanmin(train, axis=0)
    max_values = np.nanmax(train, axis=0)
    ranges = max_values - min_values
    ranges = np.where(ranges == 0, 1.0, ranges)
    train_scaled = (train - min_values) / ranges
    test_scaled = (test - min_values) / ranges
    return train_scaled.astype(np.float32), test_scaled.astype(np.float32)


def _needs_minmax_scale(train):
    return np.nanmin(train) < -1e-6 or np.nanmax(train) > 1 + 1e-6


def _load_psm(root):
    folder = _resolve_folder(root, 'PSM', 'psm')
    train_path = _try_resolve_file(folder, 'train.npy', 'PSM_train.npy')
    test_path = _try_resolve_file(folder, 'test.npy', 'PSM_test.npy')
    label_path = _try_resolve_file(folder, 'labels.npy', 'label.npy', 'PSM_test_label.npy')
    if train_path and test_path and label_path:
        train = _nan_to_float32(np.load(train_path))
        test = _nan_to_float32(np.load(test_path))
        labels = _binary_labels(np.load(label_path))
        feature_columns = [f'feature_{i}' for i in range(train.shape[1])]
        return train, test, labels, feature_columns

    train_df = pd.read_csv(_resolve_file(folder, 'train.csv'))
    test_df = pd.read_csv(_resolve_file(folder, 'test.csv'))
    label_df = pd.read_csv(_resolve_file(folder, 'test_label.csv'))
    feature_columns = list(train_df.columns[1:])
    train = _nan_to_float32(train_df.iloc[:, 1:].to_numpy())
    test = _nan_to_float32(test_df.iloc[:, 1:].to_numpy())
    labels = _binary_labels(label_df.iloc[:, -1].to_numpy())
    return train, test, labels, feature_columns


def _load_smd(root):
    folder = _resolve_folder(root, 'SMD', 'smd')
    train = _nan_to_float32(np.load(_resolve_file(folder, 'machine-1-1_train.npy', 'SMD_train.npy', 'smd_train.npy')))
    test = _nan_to_float32(np.load(_resolve_file(folder, 'machine-1-1_test.npy', 'SMD_test.npy', 'smd_test.npy')))
    labels = _binary_labels(np.load(_resolve_file(folder, 'machine-1-1_labels.npy', 'SMD_test_label.npy', 'smd_test_label.npy')))
    feature_columns = [f'feature_{i}' for i in range(train.shape[1])]
    return train, test, labels, feature_columns


def _load_swat(root):
    folder = _resolve_folder(root, 'SWAT', 'SWaT', 'swat')
    train_path = _try_resolve_file(folder, 'train.npy', 'SWaT_train.npy', 'SWAT_train.npy')
    test_path = _try_resolve_file(folder, 'test.npy', 'SWaT_test.npy', 'SWAT_test.npy')
    label_path = _try_resolve_file(folder, 'labels.npy', 'label.npy', 'SWaT_labels.npy', 'SWAT_labels.npy')
    if train_path and test_path and label_path:
        train = _nan_to_float32(np.load(train_path))
        test = _nan_to_float32(np.load(test_path))
        labels = _binary_labels(np.load(label_path))
        feature_columns = [f'feature_{i}' for i in range(train.shape[1])]
        return train, test, labels, feature_columns

    train_df = pd.read_csv(_resolve_file(folder, 'swat_train2.csv', 'SWAT_train2.csv', 'SWaT_train2.csv'))
    test_df = pd.read_csv(_resolve_file(folder, 'swat2.csv', 'SWAT2.csv', 'SWaT2.csv'))

    feature_columns = list(train_df.columns[:-1])
    train = _nan_to_float32(train_df.iloc[:, :-1].to_numpy())
    test = _nan_to_float32(test_df.iloc[:, :-1].to_numpy())
    labels = _binary_labels(test_df.iloc[:, -1].to_numpy())
    return train, test, labels, feature_columns


def _load_exathlon(root):
    folder = _resolve_folder(root, 'EXATHLON', 'Exathlon', 'exathlon')
    train = _nan_to_float32(np.load(_resolve_file(folder, 'train.npy', 'Exathlon_train.npy', 'EXATHLON_train.npy', 'exathlon_train.npy')))
    test = _nan_to_float32(np.load(_resolve_file(folder, 'test.npy', 'Exathlon_test.npy', 'EXATHLON_test.npy', 'exathlon_test.npy')))
    labels = _binary_labels(np.load(_resolve_file(folder, 'labels.npy', 'Exathlon_test_label.npy', 'EXATHLON_test_label.npy', 'exathlon_test_label.npy')))
    feature_columns = [f'feature_{i}' for i in range(train.shape[1])]
    return train, test, labels, feature_columns


LOADERS = {
    'PSM': _load_psm,
    'SMD': _load_smd,
    'SWaT': _load_swat,
    'Exathlon': _load_exathlon,
}


def _find_label_copies(test, labels):
    label_vector = labels.reshape(-1)
    matches = []
    if test.shape[0] != label_vector.shape[0]:
        return matches
    for idx in range(test.shape[1]):
        column = test[:, idx]
        if np.array_equal(column, label_vector):
            matches.append(idx)
    return matches


def check_dataset_integrity(dataset, train, test, labels, feature_columns, scaling):
    dataset_name = canonical_dataset(dataset)
    expected_sensors = DATASET_SPECS[dataset_name]['sensors']
    errors = []

    if train.ndim != 2:
        errors.append(f"train features must be 2D (time, sensors), got {train.shape}")
    if test.ndim != 2:
        errors.append(f"test features must be 2D (time, sensors), got {test.shape}")
    if labels.ndim != 2 or labels.shape[1] != 1:
        errors.append(f"labels must be point-level shape (time, 1), got {labels.shape}")
    if test.ndim == 2 and labels.ndim == 2 and test.shape[0] != labels.shape[0]:
        errors.append(f"test time length {test.shape[0]} does not match labels length {labels.shape[0]}")
    if train.ndim == 2 and train.shape[1] != expected_sensors:
        errors.append(f"train sensor dimension must be {expected_sensors}, got {train.shape[1]}")
    if test.ndim == 2 and test.shape[1] != expected_sensors:
        errors.append(f"test sensor dimension must be {expected_sensors}, got {test.shape[1]}")
    if len(feature_columns) != expected_sensors:
        errors.append(f"feature column count must be {expected_sensors}, got {len(feature_columns)}")
    if not np.isfinite(train).all():
        errors.append("train features contain NaN or Inf after loading")
    if not np.isfinite(test).all():
        errors.append("test features contain NaN or Inf after loading")

    unique_labels = np.unique(labels)
    if not set(unique_labels.tolist()).issubset({0.0, 1.0}):
        errors.append(f"labels must be binary 0/1, got values {unique_labels}")

    lower_columns = [str(col).strip().lower() for col in feature_columns]
    if dataset_name == 'PSM' and any(col.startswith('timestamp') for col in lower_columns):
        errors.append("PSM timestamp column leaked into features")
    if dataset_name == 'SWaT' and any(col == 'normal/attack' for col in lower_columns):
        errors.append("SWaT Normal/Attack label column leaked into features")

    label_copies = _find_label_copies(test, labels)
    if label_copies:
        errors.append(f"test feature column(s) exactly match point labels: {label_copies}")

    if errors:
        details = '\n'.join(f"- {error}" for error in errors)
        raise ValueError(f"{dataset_name} integrity check failed:\n{details}")

    label_values, label_counts = np.unique(labels.astype(int), return_counts=True)
    return {
        'dataset': dataset_name,
        'train_shape': train.shape,
        'test_shape': test.shape,
        'labels_shape': labels.shape,
        'sensors': test.shape[1],
        'test_time': test.shape[0],
        'scaling': scaling,
        'label_counts': {int(value): int(count) for value, count in zip(label_values, label_counts)},
    }


def load_dataset_arrays(dataset, data_root=None):
    dataset_name = canonical_dataset(dataset)
    spec = DATASET_SPECS[dataset_name]
    data_root = resolve_data_root(data_root)
    train, test, labels, feature_columns = LOADERS[dataset_name](data_root)

    scaling = spec['scaling']
    if scaling == 'auto':
        scaling = 'minmax' if _needs_minmax_scale(train) else 'none'

    if scaling == 'minmax':
        train, test = _train_minmax_scale(train, test)

    metadata = check_dataset_integrity(
        dataset_name,
        train,
        test,
        labels,
        feature_columns,
        scaling,
    )
    metadata['feature_columns'] = feature_columns
    return train, test, labels.astype(np.float32), metadata


def run_static_checks(data_root=None, datasets=None):
    data_root = resolve_data_root(data_root)
    print(f"Using data_root: {data_root}")
    dataset_names = datasets or list(DATASET_SPECS)
    summaries = []
    for dataset in dataset_names:
        train, test, labels, metadata = load_dataset_arrays(dataset, data_root)
        summary = {key: value for key, value in metadata.items() if key != 'feature_columns'}
        summaries.append(summary)
        print(
            f"{summary['dataset']}: train={summary['train_shape']} "
            f"test={summary['test_shape']} labels={summary['labels_shape']} "
            f"sensors={summary['sensors']} scaling={summary['scaling']} "
            f"labels={summary['label_counts']}"
        )
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Static checks for dataset_tranad files')
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--dataset', default='all')
    parsed = parser.parse_args()
    selected = None if parsed.dataset.lower() == 'all' else [parsed.dataset]
    run_static_checks(parsed.data_root, selected)
