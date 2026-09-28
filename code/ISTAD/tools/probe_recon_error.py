"""重建误差分布探针 — 比较 legacy 与 HGST 在测试集上的重建误差

目的：确认 F1 差距的来源
  - 若 HGST 的正常点（label=0）重建误差尾部更厚 → 解码器缺原始信号视图
    → 修复：解码器加 X̃ 跳跃连接
  - 若 HGST 的异常点（label=1）重建误差明显更低 → 异常重建太容易
    → 修复：限制表示信息 / 增强区分

用法:
    conda run -n tslib python tools/probe_recon_error.py --arch hgst \\
        --setting anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_archhgst_bs64_HGST_0 --max_batches 50
    conda run -n tslib python tools/probe_recon_error.py --arch legacy \\
        --setting anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_test_0 --max_batches 50
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from utils.tools import dotdict


def _build_args(arch, setting, gpu=0):
    base = dotdict(
        task_name="anomaly_detection",
        seq_len=96, enc_in=38, c_out=38,
        istad_kernel_size=7,
        istad_feat_gat_embed_dim=-1,
        istad_gru_n_layers=1, istad_gru_hid_dim=150,
        istad_recon_n_layers=1, istad_recon_hid_dim=150,
        istad_recon_type="kanad", istad_kanad_order=4,
        istad_dropout=0.2, istad_alpha=0.2,
        istad_n_hyperedges=-1, istad_k_top=-1,
        istad_h_param_init="normal",
        istad_branch_mode="hgat_kan_tcn", istad_tcn_type="kan",
        istad_kan_grid_size=10 if arch == "hgst" else 5,
        istad_kan_spline_order=3,
        istad_arch=arch,
        hgst_mod_scale=0.05, hgst_gate=1, hgst_use_relation=1,
        hgst_act="spline", hgst_rank=2, hgst_incidence="kan",
        hgst_s_grid_size=5, hgst_s_spline_order=3,
        hgst_fusion_norm="ln", hgst_gate_bias=0.0, hgst_decoder_input="fused",
        hgst_residual="linear", hgst_input_conv=0,
        data="SMD", root_path="./dataset/SMD", features="M",
        model_id="SMD", model="ISTAD", setting=setting,
        use_gpu=1, gpu=gpu, gpu_type="cuda",
        use_multi_gpu=False, devices="0", device_ids=[0],
        checkpoint_path=os.path.join("./checkpoints", setting, "checkpoint.pth"),
        num_workers=2, batch_size=64, itr=1, is_training=0,
        anomaly_ratio=1.0, use_bestf1_threshold=1,
        bestf1_search_mode="adaptive", bestf1_search_start=0.01,
        bestf1_search_end=2.0, bestf1_search_step_num=100,
        bestf1_coarse_step_num=200, bestf1_fine_step_num=500,
        bestf1_use_adjustment=1,
        istad_enable_explain=0, istad_explain_max_batches=0,
        istad_explain_max_points=20000, istad_explain_topk=10,
        istad_use_smoothgrad=0, istad_smoothgrad_samples=20,
        istad_smoothgrad_noise=0.1,
        istad_quadrant_threshold_mode="quantile",
        istad_quadrant_threshold_value=0.75,
        istad_quadrant_agg_error="sum", istad_quadrant_agg_saliency="sum",
        train_epochs=100, patience=5, learning_rate=0.01, lradj="type1",
        des="probe", is_training_flag=0,
    )
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["hgst", "legacy"], required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=50)
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    exp_args = _build_args(args.arch, args.setting, gpu=args.gpu)
    if not args.cuda:
        exp_args.use_gpu = 0
        exp_args.gpu_type = "cpu"

    exp = Exp_Anomaly_Detection(exp_args)
    model = exp.model
    ckpt_path = exp_args.checkpoint_path
    assert os.path.exists(ckpt_path), f"checkpoint 不存在: {ckpt_path}"
    exp._load_checkpoint(ckpt_path)
    model.eval()

    _, test_loader = exp._get_data(flag="TEST")

    errs = []      # 逐点重建误差（按窗口聚合前的 per-point）
    labs = []      # 对应标签
    per_window = []  # 每窗口聚合误差（sum over vars）
    per_window_lab = []

    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if bi >= args.max_batches:
                break
            batch_x, batch_y = batch[0], batch[1]
            batch_x = batch_x.to(exp.device)
            pred = model(batch_x, None, None, None)
            err = ((pred - batch_x) ** 2).mean(dim=-1)  # (B, L) 每点每窗口变量均值平方误差
            errs.append(err.cpu().numpy())
            labs.append(batch_y.cpu().numpy())
            per_window.append(err.cpu().numpy().reshape(-1))
            per_window_lab.append(batch_y.cpu().numpy().reshape(-1))

    errs = np.concatenate(errs).reshape(-1)
    labs = np.concatenate(labs).reshape(-1)
    assert errs.shape == labs.shape, f"{errs.shape} vs {labs.shape}"

    normal = errs[labs == 0]
    anom = errs[labs == 1]
    print(f"[probe] {args.arch}  样本 {errs.shape[0]}（正常 {normal.shape[0]} / 异常 {anom.shape[0]}）")
    print(f"  正常:  median={np.median(normal):.4f}  p95={np.percentile(normal,95):.4f}  "
          f"p99={np.percentile(normal,99):.4f}  max={normal.max():.3f}")
    print(f"  异常:  median={np.median(anom):.4f}  p95={np.percentile(anom,95):.4f}  "
          f"p99={np.percentile(anom,99):.4f}  max={anom.max():.3f}")
    # 正常/异常中位比：衡量两者可分离性
    ratio = np.median(anom) / max(np.median(normal), 1e-8)
    print(f"  异常/正常 中位比 = {ratio:.2f}")
    # 分位数处正常点占比：若正常 p99 逼近异常中位 → 尾部正常点污染
    thr_at_anom_med = np.median(anom)
    frac_norm_above = (normal > thr_at_anom_med).mean()
    print(f"  得分超过异常中位数({thr_at_anom_med:.4f})的正常点占比 = {frac_norm_above:.4%}")


if __name__ == "__main__":
    main()
