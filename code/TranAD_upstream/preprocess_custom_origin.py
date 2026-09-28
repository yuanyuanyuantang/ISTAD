import os
import numpy as np
import pandas as pd

# 定义路径
# 根据 TranAD/src/folderconstants.py，output_folder = 'processed'
# 我们将结果保存到 TranAD/processed 目录下，以匹配原有结构
INPUT_PSM_DIR = '/data/modeluse/TS/ISTAD/dataset/PSM'
INPUT_EXATHLON_DIR = '/data/modeluse/TS/ISTAD/dataset/EXATHLON'
INPUT_SMD_DIR = '/data/modeluse/TS/ISTAD/dataset/SMD'
INPUT_SWAT_DIR = '/data/modeluse/TS/ISTAD/dataset/SWAT'
OUTPUT_DIR = '/data/modeluse/TS/TranAD/processed'


def _first_existing_file(base_dir, candidates):
    """Return the first existing file path in candidates under base_dir."""
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f'None of the candidate files exist in {base_dir}: {candidates}'
    )

def sanitize_array(a, fill_values=None):
    """将 Inf 置为 NaN，并用给定列填充值或当前数组列中位数进行填补。"""
    a = np.asarray(a, dtype=float)
    a = np.where(np.isfinite(a), a, np.nan)
    if fill_values is None:
        with np.errstate(all='ignore'):
            fill_values = np.nanmedian(a, axis=0)
        # 整列全 NaN 时，nanmedian 会得到 NaN，这里回退为 0
        fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0)
    nan_mask = np.isnan(a)
    if np.any(nan_mask):
        a[nan_mask] = np.take(fill_values, np.where(nan_mask)[1])
    return a, fill_values


def normalize(a, min_a=None, max_a=None):
    """归一化到 [0, 1] 范围，带零方差保护。"""
    if min_a is None or max_a is None:
        min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    denom = max_a - min_a
    denom = np.where(denom < 1e-12, 1.0, denom)
    return (a - min_a) / denom, min_a, max_a


def load_swat_csv(path):
    """加载 SWAT CSV，提取数值特征与二值标签。"""
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
    # SWAT 常见标注为 Normal/Attack，兼容数字字符串与布尔字面量
    labels = np.where(
        raw_labels.isin(['attack', '1', 'true', 'anomaly']),
        1.0,
        0.0,
    )

    features_df = df.drop(columns=[label_col])
    features_df = features_df.apply(pd.to_numeric, errors='coerce')
    # 丢弃全 NaN 列，兼容时间戳等非数值字段
    features_df = features_df.loc[:, ~features_df.isna().all(axis=0)]

    if features_df.shape[1] == 0:
        raise ValueError(f'No numeric SWAT features found in {path}')

    return features_df.values.astype(float), labels

def process_psm():
    print("Processing PSM dataset...")
    folder = os.path.join(OUTPUT_DIR, 'PSM')
    os.makedirs(folder, exist_ok=True)
    
    # 加载数据 (跳过第一列 timestamp)
    train_df = pd.read_csv(os.path.join(INPUT_PSM_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(INPUT_PSM_DIR, 'test.csv'))
    labels_df = pd.read_csv(os.path.join(INPUT_PSM_DIR, 'test_label.csv'))
    
    train_values = train_df.values[:, 1:].astype(float)
    test_values = test_df.values[:, 1:].astype(float)
    # PSM label 原始 CSV 只有一列 (timestamp, label)
    labels_values = labels_df.values[:, 1].astype(float)

    # 使用训练集列统计填补缺失值，避免 NaN 在归一化中扩散
    train_values, fill_values = sanitize_array(train_values)
    test_values, _ = sanitize_array(test_values, fill_values)
    
    # 归一化
    train, min_a, max_a = normalize(train_values)
    test, _, _ = normalize(test_values, min_a, max_a)
    
    # TranAD 期望 label 的形状与 test 相同 (num_samples, num_features)
    # 我们将单列 label 广播到所有特征
    labels_values = np.where(np.isfinite(labels_values), labels_values, 0.0)
    if labels_values.shape[0] != test.shape[0]:
        raise ValueError(f"PSM label length mismatch: labels={labels_values.shape[0]}, test={test.shape[0]}")
    labels = np.repeat(labels_values.reshape(-1, 1), test.shape[1], axis=1)
        
    # 保存
    np.save(os.path.join(folder, 'train.npy'), train)
    np.save(os.path.join(folder, 'test.npy'), test)
    np.save(os.path.join(folder, 'labels.npy'), labels)
    print(f"PSM processed: train {train.shape}, test {test.shape}, labels {labels.shape}")

def process_exathlon():
    print("Processing Exathlon dataset...")
    folder = os.path.join(OUTPUT_DIR, 'Exathlon')
    os.makedirs(folder, exist_ok=True)
    
    # 加载数据 (Exathlon 已经是 .npy 格式)
    # 根据之前的检查: train (88230, 19), test (52844, 19), label (52844,)
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
    
    # 归一化
    train, min_a, max_a = normalize(train_raw)
    test, _, _ = normalize(test_raw, min_a, max_a)
    
    # 广播 label 形状
    if labels_raw.shape[0] != test.shape[0]:
        raise ValueError(f"Exathlon label length mismatch: labels={labels_raw.shape[0]}, test={test.shape[0]}")
    labels = np.repeat(labels_raw.reshape(-1, 1), test.shape[1], axis=1)
        
    # 保存
    np.save(os.path.join(folder, 'train.npy'), train)
    np.save(os.path.join(folder, 'test.npy'), test)
    np.save(os.path.join(folder, 'labels.npy'), labels)
    print(f"Exathlon processed: train {train.shape}, test {test.shape}, labels {labels.shape}")


def process_smd():
    print("Processing SMD dataset...")
    folder = os.path.join(OUTPUT_DIR, 'SMD')
    os.makedirs(folder, exist_ok=True)

    train_raw = np.load(os.path.join(INPUT_SMD_DIR, 'SMD_train.npy')).astype(float)
    test_raw = np.load(os.path.join(INPUT_SMD_DIR, 'SMD_test.npy')).astype(float)
    labels_raw = np.load(os.path.join(INPUT_SMD_DIR, 'SMD_test_label.npy')).astype(float)

    train_raw, fill_values = sanitize_array(train_raw)
    test_raw, _ = sanitize_array(test_raw, fill_values)
    labels_raw = np.where(np.isfinite(labels_raw), labels_raw, 0.0)

    train, min_a, max_a = normalize(train_raw)
    test, _, _ = normalize(test_raw, min_a, max_a)

    if labels_raw.shape[0] != test.shape[0]:
        raise ValueError(f"SMD label length mismatch: labels={labels_raw.shape[0]}, test={test.shape[0]}")
    labels = np.repeat(labels_raw.reshape(-1, 1), test.shape[1], axis=1)

    # Save generic names for compatibility with other scripts.
    # np.save(os.path.join(folder, 'train.npy'), train)
    # np.save(os.path.join(folder, 'test.npy'), test)
    # np.save(os.path.join(folder, 'labels.npy'), labels)

    # TranAD/main.py loads SMD files using machine-1-1_*.npy names.
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

    if labels_raw.shape[0] != test.shape[0]:
        raise ValueError(f"SWAT label length mismatch: labels={labels_raw.shape[0]}, test={test.shape[0]}")
    labels = np.repeat(labels_raw.reshape(-1, 1), test.shape[1], axis=1)

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
