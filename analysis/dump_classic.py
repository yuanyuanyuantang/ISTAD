#!/usr/bin/env python3
"""classic-7 全长分数导出：严格复刻 TranAD_improve main.py 的 test 流程。

做法：注入 sys.argv 后直接 import main.py（其 __main__ guard 不执行），复用
load_dataset / load_model / convert_to_windows / backprop 原函数——与原始
results/{MODEL}_{DATASET}/final_result.json 的产生路径逐字节一致。

注意：OmniAnomaly forward 内含 torch.rand/torch.randn_like（重参数化采样），
test 分数依赖 RNG 状态；本脚本复刻 set_seed(42) + 相同构造顺序，若存档结果
由本仓库版本 + 默认种子产生则可精确复现（回归检查见 classic_metrics.py）。

用法: python dump_classic.py <MODEL> <DATASET> <GPU>
MODEL   ∈ {DAGMM, GDN, LSTM_AD, MAD_GAN, MTAD_GAT, OmniAnomaly, TranAD}
DATASET ∈ {SMD, PSM, SWaT, Exathlon}
输出: /data/modeluse/TS/vus_diag/classic_scores/{MODEL}_{DATASET}.npz
      (score=test 逐点均值重建误差, label=逐点标签, train_score=训练集分数)
"""
import os
import sys
import json

MODEL, DATASET, GPU = sys.argv[1], sys.argv[2], sys.argv[3]
# src.constants 在 import 时依据 argv 计算 lm（TranAD 走第二组）→ 必须先注入
sys.argv = ['dump_classic.py', '--dataset', DATASET, '--model', MODEL, '--test']
os.environ.setdefault('CUDA_VISIBLE_DEVICES', GPU)

import numpy as np
import torch

TRI = '/data/modeluse/TS/TranAD_improve'
os.chdir(TRI)
sys.path.insert(0, TRI)

import main as M  # noqa: E402  (top-level 安全：仅 import 与函数定义)

WINDOWED = ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN']


def main():
    M.set_seed(42)  # main.py 默认 seed
    train_loader, test_loader, labels, metadata = M.load_dataset(DATASET)
    labelsFinal = labels.reshape(-1).astype(int)

    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    model, optimizer, scheduler, epoch, accuracy_list = M.load_model(MODEL, trainD.shape[1])
    if model.name in WINDOWED or 'TranAD' in model.name:
        trainD, testD = M.convert_to_windows(trainD, model), M.convert_to_windows(testD, model)
    model_dtype = torch.float32 if 'TranAD' in model.name else torch.float64
    if model.name == 'MTAD_GAT':
        trainD, testD = trainD.to(device=M.DEVICE, dtype=model_dtype), testD.to(dtype=model_dtype)
    else:
        trainD, testD = trainD.to(device=M.DEVICE, dtype=model_dtype), testD.to(device=M.DEVICE, dtype=model_dtype)
    trainO, testO = trainO.to(device=M.DEVICE, dtype=model_dtype), testO.to(device=M.DEVICE, dtype=model_dtype)

    model.eval()
    with torch.no_grad():
        loss, y_pred = M.backprop(0, model, testD, testO, optimizer, scheduler, training=False)
        lossT, _ = M.backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)

    lossFinal = np.mean(loss, axis=1)
    lossTfinal = np.mean(lossT, axis=1)
    assert lossFinal.shape == labelsFinal.shape, f'{lossFinal.shape} vs {labelsFinal.shape}'

    out_dir = '/data/modeluse/TS/vus_diag/classic_scores'
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{MODEL}_{DATASET}.npz')
    np.savez(out, score=lossFinal.astype(np.float64), label=labelsFinal,
             train_score=lossTfinal.astype(np.float64))

    stored = json.load(open(f'{TRI}/results/{MODEL}_{DATASET}/final_result.json'))
    print(f'[dump] {MODEL:12s} {DATASET:9s} n_test={len(lossFinal)} anom={labelsFinal.mean():.4f} '
          f'stored_f1={stored["f1"]:.4f} stored_th={stored["threshold"]:.6f} -> {out}', flush=True)


if __name__ == '__main__':
    main()
