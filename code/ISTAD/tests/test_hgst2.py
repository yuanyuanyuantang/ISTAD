"""HGST v2 最小测试套件（stdlib unittest，CPU 确定性）

覆盖（kill-switch 门槛前必须全绿）：
1. 形状：N∈{19,25,38,51} × L∈{64,96,100}，整模型 + 块级 + aux
2. 因果性：改动未来时间点，过去输出 bitwise 级不变（atol 1e-6）
3. η=0 / 无关系退化：激活器对相同 u、不同 R 输出一致
4. 初始等价：零初始化下，带 R 与不带 R 输出一致（Δw≡0）
5. R 敏感性：训练态意义下，不同 R 应产生不同输出（nonzero 调制）
6. 梯度：一次 backward 后空间样条/先验/HGAT/coeff_gen/样条系数/门控/decoder 梯度有限非零
7. 输入梯度（saliency 前提）
8. Top-K 覆盖：稀疏化后每节点至少属于一条超边
9. B 样条基 partition of unity（网格点求和≈1）
10. legacy checkpoint 严格加载回归（证明 ISTAD.py 改动未破坏 legacy）
11. setting 命名：arch=hgst2 时含 _archhgst2 标签
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.istad_layers.hgst2 import ISTADHGST2, HGST2Block
from models.istad_layers.spline_ops import BSplineBasis, ConditionalSplineActivation

torch.manual_seed(7)


def _model(n=38, l=96, **kw):
    kw.setdefault('dropout', 0.0)
    kw.setdefault('temporal_levels_max', 6)
    return ISTADHGST2(n_features=n, window_size=l, out_dim=n, **kw)


class TestShapes(unittest.TestCase):
    def test_full_model_dims(self):
        for n in (19, 25, 38, 51):
            for l in (64, 96, 100):
                m = _model(n=n, l=l)
                m.eval()
                x = torch.randn(2, l, n)
                y = m(x)
                self.assertEqual(y.shape, (2, l, n))
                self.assertTrue(torch.isfinite(y).all(), f'N={n} L={l} 非有限输出')

    def test_block_aux(self):
        blk = HGST2Block(n_features=25, seq_len=64, dropout=0.0)
        blk.eval()
        x = torch.randn(2, 64, 25)
        h, aux = blk(x, return_aux=True)
        self.assertEqual(h.shape, (2, 64, 25))
        self.assertEqual(aux['incidence'].shape[2:], (25, blk.hgat_core.n_hyperedges))
        self.assertEqual(aux['relation_state'].shape, (2, 64, 25, blk.relation_dim))
        self.assertEqual(aux['gate'].shape, (2, 64, 25))
        self.assertTrue(torch.isfinite(h).all())

    def test_param_count_smd(self):
        m = _model(n=38, l=96)
        cnt = sum(p.numel() for p in m.parameters())
        self.assertLess(cnt, 200_000, f'v2 参数量异常偏大: {cnt}')
        print(f'\n[HGST2] SMD 配置参数量: {cnt:,}')


class TestCausality(unittest.TestCase):
    def _check(self, arch_kw):
        l, n = 64, 19
        m = _model(n=n, l=l, **arch_kw)
        m.eval()
        x1 = torch.randn(1, l, n)
        x2 = x1.clone()
        t0 = l // 2
        x2[:, t0 + 1:] = torch.randn(1, l - t0 - 1, n)
        with torch.no_grad():
            y1 = m(x1)
            y2 = m(x2)
        diff = (y1[:, :t0 + 1] - y2[:, :t0 + 1]).abs().max().item()
        self.assertLess(diff, 1e-6, f'因果性违反: max diff={diff}')

    def test_full(self):
        self._check({})

    def test_full_no_spatial(self):
        self._check({'ablation': 'no_spatial'})

    def test_fixed_temporal(self):
        self._check({'ablation': 'fixed_temporal_spline'})


class TestSplineDegeneracy(unittest.TestCase):
    def test_scale0_ignores_relation(self):
        act = ConditionalSplineActivation(n_features=10, temporal_width=2,
                                          relation_dim=8, modulation_scale=0.0)
        act.eval()
        u = torch.randn(2, 20, 32)
        r1 = torch.randn(2, 32, 10, 8)
        r2 = torch.randn(2, 32, 10, 8) * 5
        with torch.no_grad():
            y1 = act(u, r1)
            y2 = act(u, r2)
        self.assertTrue(torch.allclose(y1, y2, atol=1e-6), 'scale=0 应与关系无关')

    def test_zero_init_delta_w(self):
        """零初始化下带 R 与不带 R 输出一致（Δw≡0 起步）"""
        torch.manual_seed(0)
        act = ConditionalSplineActivation(n_features=10, temporal_width=2,
                                          relation_dim=8, modulation_scale=0.05)
        act.eval()
        u = torch.randn(2, 20, 32)
        r = torch.randn(2, 32, 10, 8)
        with torch.no_grad():
            y_with = act(u, r)
            y_without = act(u, None)
        diff = (y_with - y_without).abs().max().item()
        self.assertLess(diff, 1e-6, f'零初始化后 Δw 应≡0, diff={diff}')

    def test_relation_sensitivity(self):
        """同 u 不同 R → 输出不同（调制通路真实起作用）"""
        torch.manual_seed(1)
        act = ConditionalSplineActivation(n_features=10, temporal_width=2,
                                          relation_dim=8, modulation_scale=0.05)
        for p in act.coeff_gen.parameters():
            torch.nn.init.normal_(p, std=0.5)  # 人为打破零初始化
        act.eval()
        u = torch.randn(2, 20, 32)
        r1 = torch.randn(2, 32, 10, 8)
        r2 = torch.randn(2, 32, 10, 8)
        with torch.no_grad():
            y1 = act(u, r1)
            y2 = act(u, r2)
        self.assertGreater((y1 - y2).abs().max().item(), 1e-6)


class TestGradients(unittest.TestCase):
    def test_backward_all_modules(self):
        """单次 backward 后各模块梯度有限非零。

        注：Δw = η·Σ_q A·(bases·V) 为双线性项且 coeff_gen 零初始化（A≡0），
        rank_basis 的第 0 步梯度结构性为 0（与 LoRA B=0 同理，第 1 步优化后
        coeff_gen 非零即恢复）。为验证 rank_basis 的梯度通路，先以小扰动
        打破 coeff_gen 零初始化再 backward。
        """
        torch.manual_seed(3)
        m = _model(n=25, l=64)
        m.train()
        blk = m.hgst
        for b in blk.temporal_net.blocks:
            for p in b.act.coeff_gen.parameters():
                torch.nn.init.normal_(p, std=0.1)
        blk = m.hgst
        x = torch.randn(2, 64, 25)
        y = m(x)
        y.mean().backward()

        checks = {
            'spatial a_param': blk.incidence_gen.a_param,
            'spatial c_param': blk.incidence_gen.c_param,
            'spatial prior': blk.incidence_gen.prior,
            'desc_proj': blk.desc_proj.weight,
            'var_embed': blk.var_embed.weight,
            'hgat node_transform': blk.hgat_core.node_transform.weight,
            'coeff_gen': blk.temporal_net.blocks[0].act.coeff_gen.weight,
            'temporal base_weight': blk.temporal_net.blocks[0].act.base_weight,
            'temporal rank_basis': blk.temporal_net.blocks[0].act.rank_basis,
            'gate': blk.gate.weight,
            'decoder': m.decoder[1].weight,
            'temporal conv': blk.temporal_net.blocks[0].conv.weight,
        }
        for name, p in checks.items():
            self.assertIsNotNone(p.grad, f'{name} 无梯度')
            self.assertTrue(torch.isfinite(p.grad).all(), f'{name} 梯度非有限')
            self.assertGreater(p.grad.abs().max().item(), 0, f'{name} 梯度全零')

    def test_input_grad(self):
        m = _model(n=19, l=64)
        m.eval()
        x = torch.randn(1, 64, 19, requires_grad=True)
        m(x).mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().max().item(), 0)


class TestHypergraph(unittest.TestCase):
    def test_topk_coverage(self):
        core = HGST2Block(n_features=19, seq_len=64, dropout=0.0).hgat_core
        core.eval()
        feats = torch.randn(4, 19, 16)
        logits = torch.randn(4, 19, core.n_hyperedges)
        inc = torch.softmax(logits, dim=1)
        H = core._apply_topk(inc)
        row_sum = H.sum(dim=2)  # 每节点跨超边
        self.assertTrue((row_sum > 0).all(), '存在孤立节点')

    def test_basis_partition_of_unity(self):
        b = BSplineBasis(grid_size=5, spline_order=3)
        xs = torch.tensor([-3.5, -2.0, -0.5, 0.0, 1.3, 3.9])
        bases = b(xs)
        sums = bases.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4),
                        f'partition of unity 违反: {sums}')


class TestLegacyRegression(unittest.TestCase):
    def test_legacy_checkpoint_loads(self):
        import argparse
        from models.ISTAD import Model
        ckpt = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'checkpoints', 'ISTAD',
            'anomaly_detection_SMD_ISTAD_SMD_ftM_sl96_bmhgat_kan_tcn_bs64_test_0',
            'checkpoint.pth',
        ))
        if not os.path.isfile(ckpt):
            self.skipTest('legacy SMD checkpoint 不存在')
        a = argparse.Namespace(
            task_name='anomaly_detection', seq_len=96, enc_in=38, c_out=38,
            istad_kernel_size=7, istad_feat_gat_embed_dim=-1, istad_gru_n_layers=1,
            istad_gru_hid_dim=150, istad_recon_n_layers=1, istad_recon_hid_dim=150,
            istad_recon_type='kanad', istad_kanad_order=4, istad_dropout=0.2,
            istad_alpha=0.2, istad_n_hyperedges=-1, istad_k_top=-1,
            istad_h_param_init='normal', istad_branch_mode='hgat_kan_tcn',
            istad_tcn_type='kan', istad_kan_grid_size=5, istad_kan_spline_order=3,
            istad_arch='legacy')
        m = Model(a)
        sd = torch.load(ckpt, map_location='cpu', weights_only=True)
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        m.load_state_dict(sd, strict=True)  # 不一致将抛异常

    def test_setting_name(self):
        import argparse
        from run import _build_setting
        a = argparse.Namespace(
            task_name='anomaly_detection', model_id='SMD', model='ISTAD', data='SMD',
            features='M', seq_len=96, istad_branch_mode='hgat_kan_tcn',
            istad_arch='hgst2', istad_ablation='full',
            batch_size=64, des='repro')
        s = _build_setting(a, 0)
        self.assertIn('_archhgst2', s)
        a.istad_arch = 'legacy'
        s2 = _build_setting(a, 0)
        self.assertNotIn('_arch', s2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
