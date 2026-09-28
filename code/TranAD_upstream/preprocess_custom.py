import os
import numpy as np
import pandas as pd

INPUT_PSM_DIR = '/data/modeluse/TS/ISTAD/dataset/PSM'
INPUT_EXATHLON_DIR = '/data/modeluse/TS/ISTAD/dataset/EXATHLON'
INPUT_SMD_DIR = '/data/modeluse/TS/ISTAD/dataset/SMD'
INPUT_SWAT_DIR = '/data/modeluse/TS/ISTAD/dataset/SWAT'
OUTPUT_DIR = '/data/modeluse/TS/TranAD/processed'


def _first_existing_file(base_dir, candidates):
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f'None of the candidate files exist in {base_dir}: {candidates}'
    )


def sanitize_array(a, fill_values=None):
    a = np.asarray(a, dtype=float)
    a = np.where(np.isfinite(a), a, np.nan)
    if fill_values is None:
        with np.errstate(all='ignore'):
            fill_values = np.nanmedian(a, axis=0)
        fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0)
    nan_mask = np.isnan(a)
    if np.any(nan_mask):
        a[nan_mask] = np.take(fill_values, np.where(nan_mask)[1])
    return a, fill_values


def normalize(a, min_a=None, max_a=None):
    if min_a is None or max_a is None:
        min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    denom = max_a - min_a
    denom = np.where(denom < 1e-12, 1.0, denom)
    return (a - min_a) / denom, min_a, max_a


def load_swat_csv(path):
    df = pd.read_csv(path, low_memory=False)

    label_col = None
    for col in df.columns:
        normalized = str(col).strip().lower().replace(' ', '')
        if normalized in ('normal/attack', 'label'):
            label_col = col
            break
    if label_col is None:
        raise ValueError(f'SWAT label column not found in {path}')

    raw_labels = df[label_col].astype(str).str.strip().str.lower()
    labels = np.where(
        raw_labels.isin(['attack', '1', 'true', 'anomaly']),
        1.0,
        0.0,
    )

    features_df = df.drop(columns=[label_col])
    features_df = features_df.apply(pd.to_numeric, errors='coerce')
    features_df = features_df.loc[:, ~features_df.isna().all(axis=0)]

    if features_df.shape[1] == 0:
        raise ValueError(f'No numeric SWAT features found in {path}')

    return features_df.values.astype(float), labels


def point_labels_to_feature_labels(labels_raw, test_shape, dataset_name):
    if labels_raw.shape[0] != test_shape[0]:
        raise ValueError(
            f"{dataset_name} label length mismatch: labels={labels_raw.shape[0]}, test={test_shape[0]}"
        )

    if labels_raw.ndim == 1:
        return np.repeat(labels_raw.reshape(-1, 1), test_shape[1], axis=1)

    if labels_raw.ndim == 2 and labels_raw.shape[1] == 1:
        return np.repeat(labels_raw, test_shape[1], axis=1)

    if labels_raw.ndim == 2 and labels_raw.shape[1] == test_shape[1]:
        return labels_raw

    raise ValueError(
        f"Unsupported {dataset_name} label shape: labels={labels_raw.shape}, test={test_shape}"
    )


def process_psm():
    print("Processing PSM dataset...")
    folder = os.path.join(OUTPUT_DIR, 'PSM')
    os.makedirs(folder, exist_ok=True)

    train_df = pd.read_csv(os.path.join(INPUT_PSM_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(INPUT_PSM_DIR, 'test.csv'))
    labels_df = pd.read_csv(os.path.join(INPUT_PSM_DIR, 'test_label.csv'))

    train_values = train_df.values[:, 1:].astype(float)
    test_values = test_df.values[:, 1:].astype(float)
    labels_values = labels_df.values[:, 1].astype(float)

    train_values, fill_values = sanitize_array(train_values)
    test_values, _ = sanitize_array(test_values, fill_values)

    train, min_a, max_a = normalize(train_values)
    test, _, _ = normalize(test_values, min_a, max_a)

    labels_values = np.where(np.isfinite(labels_values), labels_values, 0.0)
    labels = point_labels_to_feature_labels(labels_values, test.shape, 'PSM')

    np.save(os.path.join(folder, 'train.npy'), train)
    np.save(os.path.join(folder, 'test.npy'), test)
    np.save(os.path.join(folder, 'labels.npy'), labels)
    print(f"PSM processed: train {train.shape}, test {test.shape}, labels {labels.shape}")


def process_exathlon():
    print("Processing Exathlon dataset...")
    folder = os.path.join(OUTPUT_DIR, 'Exathlon')
    os.makedirs(folder, exist_ok=True)

    train_path = _first_existing_file(
        INPUT_EXATHLON_DIR,
        ['Exathlon_train.npy', 'SMD_train.npy'],
    )
    test_path = _first_existing_file(
        INPUT_EXATHLON_DIR,
        ['Exathlon_test.npy', 'SMD_test.npy'],
    )
    label_path = _first_existing_file(
        INPUT_EXATHLON_DIR,
        ['Exathlon_test_label.npy', 'SMD_test_label.npy'],
    )

    train_raw = np.load(train_path).astype(float)
    test_raw = np.load(test_path).astype(float)
    labels_raw = np.load(label_path).astype(float)

    train_raw, fill_values = sanitize_array(train_raw)
    test_raw, _ = sanitize_array(test_raw, fill_values)
    labels_raw = np.where(np.isfinite(labels_raw), labels_raw, 0.0)

    train, min_a, max_a = normalize(train_raw)
    test, _, _ = normalize(test_raw, min_a, max_a)

    labels = point_labels_to_feature_labels(labels_raw, test.shape, 'Exathlon')

    np.save(os.path.join(folder, 'train.npy'), train)
    np.save(os.path.join(folder, 'test.npy'), test)
    np.save(os.path.join(folder, 'labels.npy'), labels)
    print(f"Exathlon processed: train {train.shape}, test {test.shape}, labels {labels.shape}")


def process_smd():
    print("Processing SMD dataset...")
    folder = os.path.join(OUTPUT_DIR, 'SMD')
    os.makedirs(folder, exist_ok=True)

    # 与 TranAD 原始 SMD 预处理逻辑保持一致：
    # 直接读取并保存，不做 normalize，不做 sanitize_array。
    train = np.load(os.path.join(INPUT_SMD_DIR, 'SMD_train.npy')).astype(np.float64)
    test = np.load(os.path.join(INPUT_SMD_DIR, 'SMD_test.npy')).astype(np.float64)
    labels_raw = np.load(os.path.join(INPUT_SMD_DIR, 'SMD_test_label.npy')).astype(np.float64)

    print("SMD raw:", "train", train.shape, "test", test.shape, "labels", labels_raw.shape)

    labels = point_labels_to_feature_labels(labels_raw, test.shape, 'SMD')

    np.save(os.path.join(folder, 'machine-1-1_train.npy'), train)
    np.save(os.path.join(folder, 'machine-1-1_test.npy'), test)
    np.save(os.path.join(folder, 'machine-1-1_labels.npy'), labels)

    print(
        f"SMD processed: train {train.shape}, test {test.shape}, labels {labels.shape}; "
        "saved as machine-1-1_*.npy"
    )


def process_swat():
    print("Processing SWAT dataset...")
    folder = os.path.join(OUTPUT_DIR, 'SWAT')
    os.makedirs(folder, exist_ok=True)

    train_raw, _ = load_swat_csv(os.path.join(INPUT_SWAT_DIR, 'swat_train2.csv'))
    test_raw, labels_raw = load_swat_csv(os.path.join(INPUT_SWAT_DIR, 'swat2.csv'))

    train_raw, fill_values = sanitize_array(train_raw)
    test_raw, _ = sanitize_array(test_raw, fill_values)
    labels_raw = np.where(np.isfinite(labels_raw), labels_raw, 0.0)

    train, min_a, max_a = normalize(train_raw)
    test, _, _ = normalize(test_raw, min_a, max_a)

    labels = point_labels_to_feature_labels(labels_raw, test.shape, 'SWAT')

    np.save(os.path.join(folder, 'train.npy'), train)
    np.save(os.path.join(folder, 'test.npy'), test)
    np.save(os.path.join(folder, 'labels.npy'), labels)
    print(f"SWAT processed: train {train.shape}, test {test.shape}, labels {labels.shape}")


if __name__ == '__main__':
    # process_psm()
    # process_exathlon()
    process_smd()
    # process_swat()
    print("All datasets processed successfully.")