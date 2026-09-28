"""CPU tests for E2 architecture-level HGAT relation controls."""

import os
import sys
import unittest
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.ISTAD import Model
from models.istad_layers import LightweightHypergraphMixer


def _model_args(relation_mode):
    return SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=12,
        enc_in=11,
        c_out=11,
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
        istad_dropout=0.0,
        istad_dual=0,
        istad_revin=0,
        istad_evidence_head=0,
    )


def _mixer(mode, seed=87):
    torch.manual_seed(9)
    return LightweightHypergraphMixer(
        n_nodes=11,
        n_hyperedges=4,
        rank=6,
        k_top=3,
        dropout=0.0,
        relation_mode=mode,
        relation_seed=seed,
    ).eval()


class E2ArchitectureControlTests(unittest.TestCase):
    def test_dynamic_mode_matches_the_pre_e2_equations(self):
        mixer = _mixer("dynamic")
        x = torch.randn(3, 11, 1)
        with torch.no_grad():
            output, reverse, incidence = mixer(x, return_extras=True)
            state = mixer.state_norm(
                mixer.value_projection(x) + mixer.node_embedding.unsqueeze(0)
            )
            logits = torch.einsum(
                "bnr,mr->bnm", state, mixer.edge_embedding
            ) / math.sqrt(mixer.rank)
            expected_incidence = mixer._apply_topk_sparsification(
                F.softmax(logits, dim=1)
            )
            edge_state = mixer.edge_norm(torch.einsum(
                "bnm,bnr->bmr", expected_incidence, state
            ))
            expected_reverse = expected_incidence / expected_incidence.sum(
                dim=2, keepdim=True
            ).clamp_min(mixer.tau)
            message = torch.einsum(
                "bnm,bmr->bnr", expected_reverse, edge_state
            )
            expected_output = torch.sigmoid(mixer.message_gate_logit) * (
                mixer.output_projection(F.gelu(message))
            )
        self.assertTrue(torch.equal(incidence, expected_incidence))
        self.assertTrue(torch.equal(reverse, expected_reverse))
        self.assertTrue(torch.equal(output, expected_output))

    def test_dynamic_incidence_depends_on_current_values(self):
        mixer = _mixer("dynamic")
        x_a = torch.zeros(2, 11, 1)
        x_b = torch.linspace(-2.0, 2.0, 22).reshape(2, 11, 1)
        with torch.no_grad():
            _, _, incidence_a = mixer(x_a, return_extras=True)
            _, _, incidence_b = mixer(x_b, return_extras=True)
        self.assertGreater(
            torch.max(torch.abs(incidence_a - incidence_b)).item(), 1e-7
        )

    def test_static_incidence_is_trainable_but_input_independent(self):
        mixer = _mixer("static")
        with torch.no_grad():
            _, _, incidence_a = mixer(torch.randn(3, 11, 1), return_extras=True)
            _, _, incidence_b = mixer(torch.randn(3, 11, 1), return_extras=True)
        self.assertTrue(torch.allclose(incidence_a[0], incidence_a[1]))
        self.assertTrue(torch.allclose(incidence_a, incidence_b))

        mixer.train()
        output = mixer(torch.randn(3, 11, 1))
        output.square().mean().backward()
        self.assertIsNotNone(mixer.node_embedding.grad)
        self.assertIsNotNone(mixer.edge_embedding.grad)

    def test_fixed_random_control_is_balanced_sparse_and_reproducible(self):
        first = _mixer("fixed_random", seed=87)
        second = _mixer("fixed_random", seed=87)
        third = _mixer("fixed_random", seed=90)
        relation = first.fixed_random_incidence

        self.assertTrue(torch.equal(relation, second.fixed_random_incidence))
        self.assertFalse(torch.equal(relation, third.fixed_random_incidence))
        self.assertTrue(torch.allclose(
            relation.sum(dim=0), torch.ones(4), atol=1e-7
        ))
        self.assertTrue(torch.all(relation.sum(dim=1) > 0))
        self.assertLess(int(torch.count_nonzero(relation)), relation.numel())

        with torch.no_grad():
            _, _, incidence = first(torch.randn(2, 11, 1), return_extras=True)
        self.assertTrue(torch.equal(incidence[0], relation))
        self.assertTrue(torch.equal(incidence[0], incidence[1]))

    def test_no_message_returns_zero_relation_features(self):
        mixer = _mixer("no_message")
        with torch.no_grad():
            output, reverse, incidence = mixer(
                torch.randn(2, 11, 1), return_extras=True
            )
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertEqual(reverse.shape, (2, 11, 4))
        self.assertEqual(incidence.shape, (2, 11, 4))

    def test_full_models_keep_shape_and_parameter_count_constant(self):
        modes = ("dynamic", "static", "fixed_random", "no_message")
        models = []
        for mode in modes:
            torch.manual_seed(123)
            models.append(Model(_model_args(mode)))

        counts = [sum(p.numel() for p in model.parameters()) for model in models]
        self.assertEqual(len(set(counts)), 1)
        self.assertFalse(any(
            "fixed_random_incidence" in key for key in models[0].state_dict()
        ))
        reference_state = models[0].state_dict()
        for model in models[1:]:
            candidate_state = model.state_dict()
            self.assertEqual(set(reference_state), set(candidate_state))
            for key in reference_state:
                self.assertTrue(torch.equal(reference_state[key], candidate_state[key]))

        x = torch.randn(2, 12, 11)
        for model in models:
            output = model(x, None, None, None)
            self.assertEqual(output.shape, x.shape)
            self.assertTrue(torch.isfinite(output).all())

        no_message = models[-1]
        loss = no_message(x, None, None, None).square().mean()
        loss.backward()
        self.assertIsNotNone(no_message.backbone.conv.conv.weight.grad)
        self.assertIsNotNone(no_message.backbone.recon_model.network[1].weight.grad)


if __name__ == "__main__":
    unittest.main()
