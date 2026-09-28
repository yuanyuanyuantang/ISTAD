"""CPU tests for the isolated masked-channel relation bottleneck."""

import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from models.ISTAD import Model
from models.istad_layers.relation_bottleneck import (
    ChannelwiseRelationPredictionModel,
)


def _args(relation_mode="dynamic"):
    return SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=12,
        enc_in=10,
        c_out=10,
        features="M",
        seed=87,
        istad_arch="legacy",
        istad_branch_mode="hgat",
        istad_spatial_type="lite",
        istad_hgat_rank=6,
        istad_hgat_projection="linear",
        istad_hgat_relation_mode=relation_mode,
        istad_n_hyperedges=4,
        istad_k_top=3,
        istad_recon_type="pointwise",
        istad_recon_hid_dim=8,
        istad_kernel_size=3,
        istad_dropout=0.0,
        istad_dual=0,
        istad_revin=0,
        istad_evidence_head=0,
        istad_objective="reconstruct",
        istad_relation_bottleneck=1,
        istad_mask_ratio=0.2,
        istad_relation_diagnostic_batches=2,
        istad_score_mode="base_mean",
    )


def _experiment(mode="dynamic"):
    args = _args(mode)
    exp = object.__new__(Exp_Anomaly_Detection)
    exp.args = args
    exp.device = torch.device("cpu")
    exp.model = Model(args)
    return exp


class R1RelationBottleneckTests(unittest.TestCase):
    def test_architecture_removes_cross_channel_bypasses(self):
        model = Model(_args())
        self.assertEqual(model.backbone.conv.conv.groups, 10)
        self.assertIsInstance(
            model.backbone.recon_model, ChannelwiseRelationPredictionModel
        )
        x = torch.randn(2, 12, 10)
        output, auxiliary = model(x, None, None, None, return_aux=True)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(auxiliary["spatial_output"].shape, x.shape)

    def test_random_mask_has_exact_per_time_cardinality(self):
        exp = _experiment()
        x = torch.randn(3, 12, 10)
        masked, mask = exp._random_channel_mask(
            x, generator=torch.Generator().manual_seed(11)
        )
        self.assertTrue(torch.all(mask.sum(dim=-1) == 2))
        self.assertTrue(torch.equal(masked[mask], torch.zeros_like(masked[mask])))
        self.assertTrue(torch.equal(masked[~mask], x[~mask]))

    def test_deterministic_folds_cover_every_cell_once(self):
        exp = _experiment()
        x = torch.randn(2, 12, 10)
        masks = exp._deterministic_channel_masks(x)
        coverage = sum(mask.to(torch.int8) for mask in masks)
        self.assertEqual(len(masks), 5)
        self.assertTrue(torch.all(coverage == 1))
        self.assertTrue(all(torch.all(mask.sum(dim=-1) == 2) for mask in masks))

    def test_hidden_value_cannot_change_its_own_prediction(self):
        exp = _experiment()
        exp.model.eval()
        x_a = torch.randn(1, 12, 10)
        x_b = x_a.clone()
        x_b[0, 6, 4] += 100.0
        target_mask = next(
            mask for mask in exp._deterministic_channel_masks(x_a)
            if bool(mask[0, 6, 4])
        )
        with torch.no_grad():
            pred_a = exp._forward_model(x_a.masked_fill(target_mask, 0.0))
            pred_b = exp._forward_model(x_b.masked_fill(target_mask, 0.0))
        self.assertTrue(torch.equal(pred_a[0, 6, 4], pred_b[0, 6, 4]))

    def test_forward_is_causal(self):
        exp = _experiment()
        exp.model.eval()
        x_a = torch.randn(2, 12, 10)
        x_b = x_a.clone()
        x_b[:, 7:] += 25.0
        with torch.no_grad():
            output_a = exp._forward_model(x_a)
            output_b = exp._forward_model(x_b)
        self.assertTrue(torch.allclose(output_a[:, :7], output_b[:, :7], atol=1e-7))

    def test_other_channels_cannot_bypass_a_removed_hgat_message(self):
        exp = _experiment("no_message")
        exp.model.eval()
        x_a = torch.randn(2, 12, 10)
        x_b = x_a.clone()
        x_b[..., 1:] += torch.randn_like(x_b[..., 1:]) * 50.0
        with torch.no_grad():
            output_a = exp._forward_model(x_a)
            output_b = exp._forward_model(x_b)
        self.assertTrue(torch.equal(output_a[..., 0], output_b[..., 0]))

    def test_masked_loss_reaches_hgat_only_when_messages_are_enabled(self):
        dynamic = _experiment("dynamic")
        x = torch.randn(4, 12, 10)
        loss = dynamic._relation_bottleneck_loss(
            x, generator=torch.Generator().manual_seed(19)
        )
        loss.backward()
        dynamic_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in dynamic.model.backbone.feature_gat.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(dynamic_grad, 0.0)

        no_message = _experiment("no_message")
        loss = no_message._relation_bottleneck_loss(
            x, generator=torch.Generator().manual_seed(19)
        )
        loss.backward()
        no_message_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in no_message.model.backbone.feature_gat.parameters()
            if parameter.grad is not None
        )
        self.assertEqual(no_message_grad, 0.0)

    def test_diagnostics_are_finite_and_restore_relation_mode(self):
        exp = _experiment("dynamic")
        loader = [
            (torch.randn(2, 12, 10), torch.zeros(2, 12))
            for _ in range(2)
        ]
        report = exp._relation_bottleneck_diagnostics(loader)
        self.assertEqual(exp.model.backbone.feature_gat.relation_mode, "dynamic")
        self.assertGreater(report["masked_validation_mse"], 0.0)
        self.assertGreaterEqual(report["hgat_output_variance_mean"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(list(filter(
            lambda value: isinstance(value, float), report.values()
        )))).all())


if __name__ == "__main__":
    unittest.main()
