"""样条饱和/死区探针 — 诊断 HGST 时间样条是否挨饿

用法:
    conda run -n tslib python tools/probe_spline_saturation.py --arch hgst \\
        --setting anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_archhgst_bs64_HGST_0
    conda run -n tslib python tools/probe_spline_saturation.py --arch legacy \\
        --setting anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_test_0

对每个样条激活层：
  - 采集归一化后输入 x_n（进入 B 样条基的激活值）的分布统计
  - 计算超出网格区间的比例（|x_n| > grid_max → 基函数全为 0，样条分支“饱和”）
  - 比较 HGST（CausalChannelAffine 恒等起步）vs legacy（InstanceNorm，输入恒 ~N(0,1)）
    的归一化输入尺度差异——若 HGST 的 x_n 频繁落在网格外，则时间路径退化，
    只靠 SiLU 残差兜底，|Δw| 调制也失去意义。
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from models.istad_layers.spline_ops import BSplineBasis
from utils.tools import dotdict


class _GridWrap:
    """legacy 样条（grid 直接挂在模块上）的网格兼容包装"""

    def __init__(self, grid):
        self.grid = grid

    def grid_max(self):
        return float(self.grid.max().item())


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
        # 时间样条网格：HGST 训练用 grid 10/(-4,4) → K=13；legacy 用 grid 5/(-2,2) → K=8
        istad_kan_grid_size=10 if arch == "hgst" else 5,
        istad_kan_spline_order=3,
        istad_arch=arch,
        hgst_mod_scale=0.05, hgst_gate=1, hgst_use_relation=1,
        hgst_act="spline", hgst_rank=2, hgst_incidence="kan",
        hgst_s_grid_size=5, hgst_s_spline_order=3,
        hgst_fusion_norm="ln", hgst_gate_bias=0.0, hgst_decoder_input="fused",
        data="SMD", root_path="./dataset/SMD", features="M",
        model_id="SMD", model="ISTAD", setting=setting,
        use_gpu=1, gpu=gpu, gpu_type="cuda",
        use_multi_gpu=False, devices="0", device_ids=[0],
        checkpoint_path=os.path.join("./checkpoints", setting, "checkpoint.pth"),
        # exp 其余字段（_get_data 用）
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
    ap.add_argument("--max_batches", type=int, default=8)
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

    # 收集所有样条激活的 norm 模块（输入进基函数前的归一化层）。
    # HGST 用 BSplineBasis 子模块（.basis）；legacy 的 BSplineActivation 直接持有 grid buffer。
    # 统一判据：模块带 .norm 且自身拥有名为 grid 的 buffer（或持有 BSplineBasis）。
    spline_norms = []  # (name, norm_module, basis)
    for name, mod in model.named_modules():
        norm = getattr(mod, "norm", None)
        if norm is None:
            continue
        if isinstance(getattr(mod, "basis", None), BSplineBasis):
            spline_norms.append((name, norm, mod.basis))
        elif "grid" in dict(mod.named_buffers()):
            grid = mod.get_buffer("grid")
            spline_norms.append((name, norm, _GridWrap(grid)))
    print(f"[probe] 发现 {len(spline_norms)} 个样条激活层")

    # 统计容器
    stats = {i: {"count": 0, "sum_xn": 0.0, "sum_xn2": 0.0,
                 "out_grid": 0, "abs_max": 0.0} for i in range(len(spline_norms))}

    hook_handles = []

    def make_hook(i, grid_max):
        def hook(mod, inp, out):
            xn = out.detach()
            s = stats[i]
            s["count"] += xn.numel()
            s["sum_xn"] += xn.sum().item()
            s["sum_xn2"] += (xn ** 2).sum().item()
            s["out_grid"] += (xn.abs() > grid_max).sum().item()
            s["abs_max"] = max(s["abs_max"], xn.abs().max().item())
        return hook

    for i, (name, norm, basis) in enumerate(spline_norms):
        grid_max = basis.grid_max() if isinstance(basis, _GridWrap) else float(basis.grid.max().item())
        print(f"  [{i}] {name}  grid_max={grid_max}")
        handle = norm.register_forward_hook(make_hook(i, grid_max))
        hook_handles.append(handle)

    _, test_loader = exp._get_data(flag="TEST")

    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if bi >= args.max_batches:
                break
            batch = [b.to(exp.device) if torch.is_tensor(b) else b for b in batch]
            batch_x, batch_y = batch[0], batch[1]
            model(batch_x, None, None, None)
    for h in hook_handles:
        h.remove()

    print("\n[probe] 样条归一化输入分布（进基函数前的 x_n）")
    print(f"{'layer':<24}{'mean':>9}{'std':>9}{'|x|>grid':>11}{'abs_max':>10}")
    for i, (name, norm, basis) in enumerate(spline_norms):
        s = stats[i]
        if s["count"] == 0:
            continue
        mean = s["sum_xn"] / s["count"]
        std = np.sqrt(max(0.0, s["sum_xn2"] / s["count"] - mean ** 2))
        frac = s["out_grid"] / s["count"]
        grid_max = basis.grid_max() if isinstance(basis, _GridWrap) else float(basis.grid.max().item())
        print(f"{name:<24}{mean:>9.4f}{std:>9.4f}{frac:>10.4%}{s['abs_max']:>10.4f}")
        if frac > 0.05:
            print(f"    ⚠ 饱和率 {frac:.2%} > 5%：该层 {name} 样条频繁落在网格外（基函数全 0），"
                  f"退化为仅 SiLU 残差分支")


if __name__ == "__main__":
    main()
