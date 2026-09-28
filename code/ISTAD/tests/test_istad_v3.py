"""CPU smoke tests for ISTAD-v3 denoising and evidence fusion."""

import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_provider.data_loader import (
    MSLSegLoader,
    SMAPSegLoader,
    SMDSegLoader,
    _parts_from_lengths,
    _append_entity_context,
    _scale_segmented_train_val,
    _scale_train_val,
    _train_val_split,
    _window_starts,
)
from data_provider import data_loader as loader_module
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from models.ISTAD import Model
from models.istad_layers import (
    BoundedResidualKANProjection,
    LightweightHypergraphMixer,
    SharedKANProjection,
)
from utils.innovation import (
    TrainingOnlyInnovationScorer,
    hypergraph_pool_feature_evidence,
    rank_safe_hgat_refine,
    training_ecdf_recalibrate,
    training_only_hgat_reliability_gate,
)
from utils.event_metrics import event_overlap_metrics, event_ranges
from utils.causal_prior import EntitywiseComponentECDF, fit_signed_causal_prior
from utils.tools import adjustment, bf_search_adaptive


def _args(**updates):
    values = dict(
        task_name='anomaly_detection', seq_len=16, enc_in=5, c_out=5,
        features='M', istad_arch='legacy', istad_branch_mode='kan_tcn',
        istad_tcn_type='kan', istad_recon_type='kanad', istad_kanad_order=2,
        istad_kernel_size=3, istad_dropout=0.0, istad_dual=1,
        istad_dual_lambda=1.0, istad_revin=0, istad_evidence_head=1,
        istad_denoise=1, istad_clean_lambda=1.0, istad_denoise_lambda=1.0,
        istad_evidence_lambda=0.2, istad_point_evidence_lambda=0.5,
        istad_corrupt_prob=1.0,
        istad_corrupt_time_ratio=0.25, istad_corrupt_channel_ratio=0.4,
        istad_corrupt_scale=1.5, istad_score_mode='evidence',
        istad_score_aggregate='learned', istad_evidence_transform='prob',
        istad_score_topk=2, seed=7,
    )
    values.update(updates)
    return SimpleNamespace(**values)


class TestEvidenceHead(unittest.TestCase):
    def test_shapes_positive_joint_response_and_gradient(self):
        args = _args()
        model = Model(args)
        model.eval()
        x = torch.randn(2, 16, 5)
        outputs = model(x, None, None, None)
        self.assertEqual(outputs.shape, (2, 16, 10))
        logits = model.evidence_logits(x, outputs)
        self.assertEqual(logits.shape, x.shape)

        base = outputs[..., :5].detach().clone()
        revin = outputs[..., 5:].detach().clone()
        low = model.evidence_head(torch.zeros_like(base), torch.zeros_like(revin))
        high = model.evidence_head(torch.ones_like(base), torch.ones_like(revin))
        self.assertTrue(torch.all(high > low))

        exp = object.__new__(Exp_Anomaly_Detection)
        exp.args = args
        exp.model = model
        model.train()
        loss, clean_loss, denoise_loss, evidence_loss = exp._v3_loss(x, nn.MSELoss())
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(clean_loss.item(), 0.0)
        self.assertGreater(denoise_loss.item(), 0.0)
        self.assertGreater(evidence_loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(model.evidence_head.raw_weight.grad)
        self.assertGreater(model.evidence_head.raw_weight.grad.abs().max().item(), 0.0)
        self.assertIsNotNone(model.evidence_head.pool_logit.grad)
        self.assertGreater(model.evidence_head.pool_logit.grad.abs().item(), 0.0)

    def test_corruption_mask_and_point_score(self):
        args = _args()
        model = Model(args)
        exp = object.__new__(Exp_Anomaly_Detection)
        exp.args = args
        exp.model = model
        x = torch.randn(4, 16, 5)
        generator = torch.Generator().manual_seed(11)
        corrupted, mask = exp._synthetic_corrupt(x, generator=generator)
        self.assertEqual(mask.shape, x.shape)
        self.assertTrue(mask.any())
        self.assertTrue(torch.all((corrupted != x)[mask]))
        outputs = model(x, None, None, None)
        score = exp._point_score(x, outputs)
        self.assertEqual(score.shape, (4, 16))
        self.assertTrue(torch.isfinite(score).all())
        components = exp._calibration_components(x, outputs)
        self.assertEqual(components.shape, (4, 16, 2))
        self.assertTrue(torch.isfinite(components).all())


class TestTargetForecastObjective(unittest.TestCase):
    def test_target_is_lagged_context_is_current_and_output_is_compact(self):
        args = _args(
            c_out=1,
            istad_dual=0,
            istad_objective='target_forecast',
            istad_target_features='0',
            istad_forecast_lag=1,
            istad_corrupt_target_only=1,
            istad_score_aggregate='calibrated',
            istad_score_topk=1,
        )
        model = Model(args)
        exp = object.__new__(Exp_Anomaly_Detection)
        exp.args = args
        exp.model = model

        x = torch.arange(2 * 16 * 5, dtype=torch.float32).reshape(2, 16, 5) / 10.0
        model_input = exp._model_input(x)
        self.assertTrue(torch.equal(model_input[:, 1:, 0], x[:, :-1, 0]))
        self.assertTrue(torch.equal(model_input[:, :, 1:], x[:, :, 1:]))
        self.assertTrue(torch.equal(model_input[:, 0, 0], torch.zeros(2)))

        outputs = exp._forward_model(x)
        self.assertEqual(outputs.shape, (2, 16, 1))
        logits = model.evidence_logits(x, outputs)
        self.assertEqual(logits.shape, (2, 16, 1))
        components = exp._calibration_components(x, outputs)
        self.assertEqual(components.shape, (2, 16, 2))
        self.assertTrue(torch.equal(components[:, 0], torch.zeros(2, 2)))

    def test_target_only_corruption_and_loss_have_gradients(self):
        args = _args(
            c_out=1,
            istad_dual=0,
            istad_objective='target_forecast',
            istad_target_features='0',
            istad_forecast_lag=1,
            istad_corrupt_target_only=1,
            istad_score_topk=1,
        )
        model = Model(args)
        exp = object.__new__(Exp_Anomaly_Detection)
        exp.args = args
        exp.model = model
        x = torch.randn(4, 16, 5)
        corrupted, mask = exp._synthetic_corrupt(
            x, generator=torch.Generator().manual_seed(19)
        )
        self.assertTrue(mask[..., 0].any())
        self.assertFalse(mask[..., 1:].any())
        self.assertTrue(torch.equal(corrupted[..., 1:], x[..., 1:]))

        loss, clean, denoise, evidence = exp._v3_loss(x, nn.MSELoss())
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(clean.item(), 0.0)
        self.assertGreater(denoise.item(), 0.0)
        self.assertGreater(evidence.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(model.backbone.recon_model.input_proj.weight.grad)
        self.assertGreater(
            model.backbone.recon_model.input_proj.weight.grad.abs().max().item(), 0.0
        )

    def test_standard_tcn_pointwise_forecaster_has_no_future_leakage(self):
        args = _args(
            c_out=1,
            istad_dual=0,
            istad_objective='target_forecast',
            istad_target_features='0',
            istad_forecast_lag=1,
            istad_corrupt_target_only=1,
            istad_branch_mode='tcn',
            istad_tcn_type='standard',
            istad_recon_type='pointwise',
            istad_recon_hid_dim=32,
            istad_evidence_head=0,
        )
        model = Model(args).eval()
        exp = object.__new__(Exp_Anomaly_Detection)
        exp.args = args
        exp.model = model
        original = torch.randn(2, 16, 5)
        changed_future = original.clone()
        changed_future[:, 10:] += 100.0 * torch.randn_like(changed_future[:, 10:])
        with torch.no_grad():
            before = exp._forward_model(original)
            after = exp._forward_model(changed_future)
        self.assertTrue(torch.allclose(before[:, :10], after[:, :10], atol=1e-6))


class TestV7CausalPriorHypergraph(unittest.TestCase):
    def _v7_args(self, **updates):
        values = dict(
            c_out=1,
            istad_arch='v7',
            istad_dual=0,
            istad_revin=0,
            istad_denoise=0,
            istad_evidence_head=0,
            istad_objective='target_forecast',
            istad_target_features='0',
            istad_forecast_lag=1,
            istad_recon_hid_dim=16,
            istad_dropout=0.0,
            istad_v7_relation_dim=8,
            istad_v7_prior_topk=3,
            istad_v7_prior_strength=1.0,
            istad_v7_prior_floor=0.01,
            istad_v7_prior_lambda=0.05,
            istad_v7_relation_score_weight=0.25,
        )
        values.update(updates)
        return _args(**values)

    def test_prior_fitting_respects_entities_and_recovers_context_source(self):
        rng = np.random.default_rng(41)
        context = rng.normal(size=80)
        target = 2.5 * context + rng.normal(scale=0.02, size=80)
        points = np.column_stack([target, context, rng.normal(size=80)])
        entities = np.repeat([0, 1], 40)
        prior, signs, coefficients, metadata = fit_signed_causal_prior(
            points, [0], lag=1, ridge=1e-2, topk=2, entity_ids=entities
        )
        self.assertEqual(metadata['n_pairs'], 78)
        self.assertEqual(prior.shape, (3, 1))
        self.assertAlmostEqual(float(prior[:, 0].sum()), 1.0, places=6)
        self.assertGreater(abs(float(coefficients[1, 0])), 2.0)
        self.assertGreater(float(prior[1, 0]), float(prior[0, 0]))
        self.assertGreater(float(signs[1, 0]), 0.0)

    def test_entity_ecdf_is_train_only_and_entity_specific(self):
        train = np.array([
            [0.0, 10.0], [1.0, 20.0], [100.0, 1000.0], [200.0, 2000.0]
        ])
        entities = np.array([0, 0, 1, 1])
        test = np.array([[0.5, 15.0], [150.0, 1500.0]])
        test_entities = np.array([0, 1])
        calibrator = EntitywiseComponentECDF().fit(train, entities)
        calibrated = calibrator.transform(test, test_entities)
        extended = calibrator.transform(
            np.vstack([test, [[-1e9, 1e9]]]), np.array([0, 1, 0])
        )
        np.testing.assert_allclose(calibrated, extended[:2])
        np.testing.assert_allclose(calibrated[0], calibrated[1])

    def test_v7_is_causal_and_hypergraph_has_prediction_gradient(self):
        args = self._v7_args()
        model = Model(args).eval()
        prior = np.array([[0.6], [0.3], [0.1], [0.0], [0.0]], dtype=np.float32)
        signs = np.array([[1.0], [1.0], [-1.0], [1.0], [1.0]], dtype=np.float32)
        model.set_causal_prior(prior, signs)
        exp = object.__new__(Exp_Anomaly_Detection)
        exp.args = args
        exp.model = model

        original = torch.randn(2, 16, 5)
        changed_future = original.clone()
        changed_future[:, 10:] += 100.0 * torch.randn_like(changed_future[:, 10:])
        with torch.no_grad():
            before, before_aux = exp._forward_model(original, return_aux=True)
            after, after_aux = exp._forward_model(changed_future, return_aux=True)
        self.assertTrue(torch.allclose(before[:, :10], after[:, :10], atol=1e-6))
        self.assertTrue(torch.allclose(
            before_aux['incidence'][:, :10], after_aux['incidence'][:, :10], atol=1e-6
        ))
        self.assertIn('backbone.causal_prior', model.state_dict())
        self.assertTrue(bool(model.backbone.causal_prior_ready.item()))

        model.train()
        loss, forecast, prior_loss = exp._v7_loss(original, nn.MSELoss())
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(forecast.item(), 0.0)
        self.assertTrue(torch.isfinite(prior_loss))
        loss.backward()
        self.assertIsNotNone(model.backbone.target_query.grad)
        self.assertGreater(model.backbone.target_query.grad.abs().max().item(), 0.0)

    def test_v71_graph_residual_is_nested_and_learns_gates(self):
        full_args = self._v7_args(
            istad_arch='v71', istad_v7_relation_score_weight=0.0,
            istad_v71_graph_gate_init=0.05,
            istad_v71_prior_gate_init=0.5,
        )
        temporal_args = self._v7_args(
            istad_arch='v71', istad_v7_use_hypergraph=0,
            istad_v7_prior_strength=0.0, istad_v7_prior_lambda=0.0,
            istad_v7_relation_score_weight=0.0,
            istad_v71_graph_gate_init=0.05,
            istad_v71_prior_gate_init=0.5,
        )
        torch.manual_seed(73)
        full = Model(full_args)
        torch.manual_seed(73)
        temporal = Model(temporal_args)
        prior = np.array([[0.6], [0.3], [0.1], [0.0], [0.0]], dtype=np.float32)
        full.set_causal_prior(prior)

        x = torch.randn(2, 16, 5)
        full.backbone.graph_gate_logit.data.fill_(-100.0)
        full.eval()
        temporal.eval()
        with torch.no_grad():
            full_prediction, auxiliary = full(
                x, None, None, None, return_aux=True
            )
            temporal_prediction = temporal(x, None, None, None)
        self.assertTrue(torch.allclose(
            full_prediction, temporal_prediction, atol=1e-6
        ))
        self.assertTrue(torch.all(auxiliary['prior_gate'] > 0.0))
        self.assertTrue(torch.all(auxiliary['prior_gate'] < 1.0))

        full.backbone.graph_gate_logit.data.fill_(-2.9444389791664403)
        full.train()
        exp = object.__new__(Exp_Anomaly_Detection)
        exp.args = full_args
        exp.model = full
        loss, _, _ = exp._v7_loss(x, nn.MSELoss())
        loss.backward()
        self.assertIsNotNone(full.backbone.graph_gate_logit.grad)
        self.assertTrue(torch.isfinite(full.backbone.graph_gate_logit.grad).all())
        self.assertIsNotNone(full.backbone.prior_gate_logit.grad)
        self.assertTrue(torch.isfinite(full.backbone.prior_gate_logit.grad).all())

    def test_entity_context_nodes_are_constant_within_each_entity(self):
        values = np.arange(18, dtype=np.float32).reshape(9, 2)
        augmented = _append_entity_context(values, (4, 5))
        self.assertEqual(augmented.shape, (9, 4))
        np.testing.assert_array_equal(augmented[:4, 2:], [[1.0, 0.0]] * 4)
        np.testing.assert_array_equal(augmented[4:, 2:], [[0.0, 1.0]] * 5)

    def test_smd_loader_appends_entity_context_without_crossing_boundaries(self):
        with tempfile.TemporaryDirectory() as root:
            train = np.arange(34, dtype=np.float32).reshape(17, 2)
            test = np.arange(26, dtype=np.float32).reshape(13, 2)
            label = np.zeros(13, dtype=np.float32)
            np.save(os.path.join(root, 'SMD_train.npy'), train)
            np.save(os.path.join(root, 'SMD_test.npy'), test)
            np.save(os.path.join(root, 'SMD_test_label.npy'), label)
            args = SimpleNamespace(
                istad_entity_aware=1,
                istad_entity_context=1,
                istad_holdout_val=1,
                enc_in=4,
            )
            with mock.patch.object(loader_module, 'SMD_TRAIN_LENGTHS', (10, 7)), \
                    mock.patch.object(loader_module, 'SMD_TEST_LENGTHS', (7, 6)):
                dataset = SMDSegLoader(
                    args, root, win_size=2, step=2, flag='test'
                )
            self.assertEqual(dataset.train.shape[1], 4)
            self.assertEqual(dataset.val.shape[1], 4)
            self.assertEqual(dataset.test.shape[1], 4)
            np.testing.assert_array_equal(dataset.test[:7, 2:], [[1.0, 0.0]] * 7)
            np.testing.assert_array_equal(dataset.test[7:, 2:], [[0.0, 1.0]] * 6)
            starts = dataset.window_starts
            self.assertFalse(np.any((starts < 7) & (starts + 2 > 7)))


class TestStrictHoldout(unittest.TestCase):
    def test_opt_in_split_is_disjoint(self):
        values = np.arange(100)
        train, val = _train_val_split(SimpleNamespace(istad_holdout_val=1), values)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 20)
        self.assertEqual(train[-1], 79)
        self.assertEqual(val[0], 80)

    def test_legacy_split_is_preserved(self):
        values = np.arange(100)
        train, val = _train_val_split(SimpleNamespace(istad_holdout_val=0), values)
        self.assertEqual(len(train), 100)
        self.assertEqual(val[0], 80)

    def test_strict_scaler_uses_training_partition_only(self):
        from sklearn.preprocessing import StandardScaler

        values = np.concatenate([np.zeros((80, 1)), np.full((20, 1), 100.0)])
        train, val = _scale_train_val(
            SimpleNamespace(istad_holdout_val=1), StandardScaler(), values
        )
        self.assertAlmostEqual(float(train.mean()), 0.0)
        self.assertGreater(float(val.mean()), 90.0)

    def test_segmented_split_and_windows_do_not_cross_entities(self):
        from sklearn.preprocessing import StandardScaler

        values = np.arange(34, dtype=np.float32).reshape(17, 2)
        parts = _parts_from_lengths(values, (10, 7))
        self.assertEqual([len(part) for part in parts], [10, 7])
        train, val, train_lengths, val_lengths = _scale_segmented_train_val(
            StandardScaler(), values, (10, 7)
        )
        self.assertEqual(train_lengths, (8, 5))
        self.assertEqual(val_lengths, (2, 2))
        self.assertEqual(len(train) + len(val), len(values))
        starts = _window_starts((10, 7), win_size=4, step=3)
        self.assertEqual(starts.tolist(), [0, 3, 6, 10, 13])
        self.assertTrue(all(not (start < 10 < start + 4) for start in starts))

    def test_nasa_loaders_isolate_entity_windows(self):
        original_lengths = {
            name: getattr(loader_module, name)
            for name in (
                'MSL_TRAIN_LENGTHS', 'MSL_TEST_LENGTHS',
                'SMAP_TRAIN_LENGTHS', 'SMAP_TEST_LENGTHS',
            )
        }
        try:
            for name in original_lengths:
                setattr(loader_module, name, (10, 7))
            args = SimpleNamespace(istad_holdout_val=1, istad_entity_aware=1)
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                for dataset_name, loader_class in (
                    ('MSL', MSLSegLoader), ('SMAP', SMAPSegLoader)
                ):
                    dataset_root = root / dataset_name
                    dataset_root.mkdir()
                    train = np.arange(34, dtype=np.float64).reshape(17, 2)
                    test = train + 100.0
                    labels = np.zeros(17, dtype=np.int8)
                    for part, values in (
                        ('train', train), ('test', test), ('test_label', labels)
                    ):
                        np.save(dataset_root / f'{dataset_name}_{part}.npy', values)

                    dataset = loader_class(
                        args, str(dataset_root), win_size=4, step=4, flag='test'
                    )
                    with self.subTest(dataset=dataset_name):
                        self.assertEqual(dataset.window_starts.tolist(), [0, 4, 10])
                        self.assertEqual(len(dataset), 3)
                        self.assertEqual(dataset.point_entity_ids.tolist(), [0] * 8 + [1] * 4)
                        for start in dataset.window_starts:
                            self.assertFalse(start < 10 < start + dataset.win_size)
        finally:
            for name, lengths in original_lengths.items():
                setattr(loader_module, name, lengths)


class TestTrainingOnlyInnovationScorer(unittest.TestCase):
    def test_fused_recalibration_uses_only_training_reference(self):
        train = np.array([0.1, 0.3, 0.2, 0.3])
        test = np.array([0.05, 0.2, 0.3, 0.9])
        calibrated_train, calibrated_test = training_ecdf_recalibrate(
            train, test
        )

        np.testing.assert_allclose(
            calibrated_train, np.array([0.25, 1.0, 0.5, 1.0])
        )
        np.testing.assert_allclose(
            calibrated_test, np.array([0.0, 0.5, 1.0, 1.0])
        )
        # Appending arbitrary evaluation values must not change existing ranks.
        _, extended = training_ecdf_recalibrate(
            train, np.concatenate([test, np.array([-100.0, 100.0])])
        )
        np.testing.assert_allclose(extended[:len(test)], calibrated_test)

    def test_sparse_score_is_finite_and_responds_to_a_spike(self):
        rng = np.random.default_rng(17)
        train = np.zeros((500, 4), dtype=np.float64)
        for index in range(1, len(train)):
            train[index] = 0.8 * train[index - 1] + rng.normal(0.0, 0.2, 4)

        scorer = TrainingOnlyInnovationScorer(lag=1, pool='auto').fit(train)
        test = train[:100].copy()
        test[60, 2] += 8.0
        score = scorer.score(test)
        tail_score = scorer.score(test, tail_transform=True)

        self.assertEqual(scorer.metadata.selected_pool, 'sparse')
        self.assertEqual(scorer.metadata.coefficient_count, 16)
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(np.isfinite(tail_score).all())
        self.assertGreater(score[60], np.quantile(score[:50], 0.99))

    def test_feature_evidence_exactly_recovers_v4_raw_pooling(self):
        rng = np.random.default_rng(29)
        train = rng.normal(size=(300, 5))
        for pool in ('sparse', 'dense'):
            scorer = TrainingOnlyInnovationScorer(pool=pool).fit(train)
            evidence = scorer.feature_evidence(train)
            recovered = (
                evidence.max(axis=1)
                if pool == 'sparse'
                else evidence.mean(axis=1)
            )
            np.testing.assert_allclose(recovered, scorer.raw_score(train))

    def test_signed_innovation_retains_direction_and_entity_boundaries(self):
        rng = np.random.default_rng(31)
        train = rng.normal(size=(300, 3))
        scorer = TrainingOnlyInnovationScorer(pool='sparse').fit(train)
        signed = scorer.signed_feature_innovation(train)
        _, residual = scorer._residuals(train)
        np.testing.assert_allclose(
            signed[1:], residual / scorer.residual_scale_[None, :]
        )
        self.assertTrue(np.all(signed[1:][residual > 0] > 0))
        self.assertTrue(np.all(signed[1:][residual < 0] < 0))

        entities = np.repeat([0, 1], 150)
        bounded = scorer.signed_feature_innovation(train, entities)
        np.testing.assert_array_equal(bounded[[0, 150]], 0.0)

    def test_hgat_pool_and_train_only_kill_switch(self):
        evidence = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        incidence = np.array([
            [[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
            [[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
        ])
        sparse = hypergraph_pool_feature_evidence(
            evidence, incidence, pool='sparse'
        )
        dense = hypergraph_pool_feature_evidence(
            evidence, incidence, pool='dense'
        )
        np.testing.assert_allclose(sparse, [2.5, 5.5])
        np.testing.assert_allclose(dense, [2.0, 5.0])

        base = np.tile(np.linspace(0.0, 1.0, 20), 5)
        graph = base.copy()
        graph[80:] = 100.0
        weight, metadata = training_only_hgat_reliability_gate(
            base, graph, max_weight=0.2
        )
        self.assertEqual(weight, 0.0)
        self.assertTrue(metadata['kill_switch'])

    def test_rank_safe_hgat_refinement_cannot_invert_v4_ranks(self):
        calibration_size = 99
        base = np.array([0.10, 0.11, 0.11, 0.12])
        graph = np.array([1.0, 1.0, 0.0, 0.0])
        refined, epsilon = rank_safe_hgat_refine(
            base, graph, calibration_size
        )
        self.assertEqual(epsilon, 0.005)
        self.assertLess(refined[0], refined[1])
        self.assertGreater(refined[1], refined[2])
        self.assertLess(refined[2], refined[3])

    def test_auto_pool_falls_back_when_training_channels_are_degenerate(self):
        rng = np.random.default_rng(23)
        varying = rng.normal(size=(300, 4))
        constant = np.zeros((300, 2))
        train = np.concatenate([varying, constant], axis=1)
        scorer = TrainingOnlyInnovationScorer(pool='auto').fit(train)

        self.assertGreaterEqual(scorer.metadata.degenerate_fraction, 2 / 6)
        self.assertEqual(scorer.metadata.selected_pool, 'dense')

    def test_lagged_pairs_never_cross_an_entity_boundary(self):
        points = np.arange(18, dtype=np.float64).reshape(9, 2)
        entity_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        scorer = TrainingOnlyInnovationScorer(lag=2)
        valid, _, _ = scorer._lagged(points, entity_ids)

        self.assertEqual(valid.tolist(), [2, 3, 6, 7, 8])
        scorer.fit(points, entity_ids)
        score = scorer.score(points, entity_ids)
        self.assertLess(score[4], 0.5)
        self.assertLess(score[5], 0.5)

    def test_point_adjustment_and_search_respect_entity_boundaries(self):
        # The anomaly at the end of entity 0 must not propagate into entity 1.
        labels = np.array([0, 1, 1, 1, 1, 0])
        prediction = np.array([0, 0, 1, 0, 0, 0])
        entities = np.array([0, 0, 0, 1, 1, 1])
        _, adjusted = adjustment(labels, prediction, entity_ids=entities)
        self.assertEqual(adjusted.tolist(), [0, 1, 1, 0, 0, 0])

        result = bf_search_adaptive(
            score=np.array([0.0, 0.2, 0.9, 0.2, 0.2, 0.0]),
            label=labels,
            coarse_step_num=10,
            fine_step_num=20,
            verbose=False,
            use_adjustment=True,
            entity_ids=entities,
        )
        self.assertGreater(result['f1'], 0.6)
        self.assertLess(result['f1'], 1.0)


class TestEventMetrics(unittest.TestCase):
    def test_event_ranges_and_overlap_respect_entity_boundaries(self):
        labels = np.array([0, 1, 1, 1, 1, 0])
        prediction = np.array([0, 0, 1, 0, 1, 0])
        entities = np.array([0, 0, 0, 1, 1, 1])
        self.assertEqual(event_ranges(labels, entities), ((1, 3), (3, 5)))
        metrics = event_overlap_metrics(labels, prediction, entities)
        self.assertEqual(metrics['true_events'], 2)
        self.assertEqual(metrics['predicted_events'], 2)
        self.assertEqual(metrics['detected_events'], 2)
        self.assertEqual(metrics['event_f1'], 1.0)
        self.assertAlmostEqual(metrics['mean_detection_delay'], 1.0)

    def test_fragmented_predictions_are_penalized_by_event_precision(self):
        labels = np.array([0, 1, 1, 1, 1, 1, 0, 0])
        prediction = np.array([1, 1, 0, 1, 0, 1, 0, 1])
        metrics = event_overlap_metrics(labels, prediction)
        self.assertEqual(metrics['true_events'], 1)
        self.assertEqual(metrics['predicted_events'], 4)
        self.assertEqual(metrics['correct_predicted_events'], 3)
        self.assertAlmostEqual(metrics['event_precision'], 0.75)
        self.assertAlmostEqual(metrics['event_recall'], 1.0)
        self.assertAlmostEqual(metrics['event_f1'], 6.0 / 7.0)


class TestLightweightHypergraphMixer(unittest.TestCase):
    def test_bounded_residual_kan_starts_as_linear_and_learns(self):
        projection = BoundedResidualKANProjection(out_features=8)
        x = torch.randn(4, 19, 1, requires_grad=True)
        linear = projection.linear(x)
        output = projection(x)
        self.assertTrue(torch.equal(output, linear))
        self.assertEqual(sum(p.numel() for p in projection.parameters()), 72)
        output.square().mean().backward()
        self.assertGreater(projection.spline_weight.grad.abs().max().item(), 0.0)

    def test_shared_kan_projection_is_small_and_differentiable(self):
        projection = SharedKANProjection(
            out_features=8, grid_size=5, spline_order=3
        )
        x = torch.randn(4, 19, 1, requires_grad=True)
        output = projection(x)
        self.assertEqual(output.shape, (4, 19, 8))
        self.assertEqual(sum(p.numel() for p in projection.parameters()), 80)
        output.square().mean().backward()
        self.assertGreater(x.grad.abs().max().item(), 0.0)
        self.assertTrue(all(parameter.grad is not None for parameter in projection.parameters()))

    def test_kan_hgat_keeps_the_relation_interface(self):
        mixer = LightweightHypergraphMixer(
            n_nodes=19, n_hyperedges=9, rank=8, k_top=3, dropout=0.0,
            projection_type='kan', kan_grid_size=5, kan_spline_order=3,
        )
        output, reverse, incidence = mixer(
            torch.randn(4, 19, 1), return_extras=True
        )
        self.assertEqual(output.shape, (4, 19, 1))
        self.assertEqual(incidence.shape, (4, 19, 9))
        self.assertTrue(torch.allclose(
            incidence.sum(dim=1), torch.ones(4, 9), atol=1e-6
        ))
        self.assertTrue(torch.allclose(
            reverse.sum(dim=2), torch.ones(4, 19), atol=1e-6
        ))

    def test_shape_sparse_coverage_and_gradients(self):
        mixer = LightweightHypergraphMixer(
            n_nodes=19,
            n_hyperedges=9,
            rank=8,
            k_top=3,
            dropout=0.0,
        )
        x = torch.randn(4, 19, 1, requires_grad=True)
        output, reverse, incidence = mixer(x, return_extras=True)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(incidence.shape, (4, 19, 9))
        self.assertEqual(reverse.shape, incidence.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue((incidence.sum(dim=2) > 0).all())
        self.assertTrue(torch.allclose(
            incidence.sum(dim=1), torch.ones(4, 9), atol=1e-6
        ))
        self.assertTrue(torch.allclose(
            reverse.sum(dim=2), torch.ones(4, 19), atol=1e-6
        ))

        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().max().item(), 0.0)
        for parameter in mixer.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_full_model_is_compatible_and_spatial_branch_is_smaller(self):
        common = dict(
            seq_len=16,
            enc_in=51,
            c_out=51,
            istad_branch_mode='hgat_kan_tcn',
            istad_feat_gat_embed_dim=256,
            istad_n_hyperedges=20,
            istad_k_top=10,
            istad_dual=0,
        )
        legacy = Model(_args(**common, istad_spatial_type='legacy'))
        lite = Model(_args(
            **common, istad_spatial_type='lite', istad_hgat_rank=16
        ))
        x = torch.randn(2, 16, 51)
        self.assertEqual(lite(x, None, None, None).shape, x.shape)

        def spatial_parameters(model):
            modules = (
                model.backbone.node_project_in,
                model.backbone.feature_gat,
                model.backbone.node_project_out,
            )
            return sum(
                parameter.numel()
                for module in modules
                for parameter in module.parameters()
            )

        legacy_count = spatial_parameters(legacy)
        lite_count = spatial_parameters(lite)
        self.assertLess(lite_count, legacy_count * 0.02)

    def test_full_model_returns_dynamic_incidence_for_integrated_scoring(self):
        model = Model(_args(
            istad_branch_mode='hgat',
            istad_spatial_type='lite',
            istad_hgat_rank=8,
            istad_n_hyperedges=3,
            istad_k_top=2,
            istad_dual=0,
            istad_revin=0,
            istad_evidence_head=0,
            istad_recon_type='pointwise',
            istad_recon_hid_dim=8,
        )).eval()
        x = torch.randn(2, 16, 5)
        with torch.no_grad():
            reconstruction, auxiliary = model(
                x, None, None, None, return_aux=True
            )
        self.assertEqual(reconstruction.shape, x.shape)
        self.assertEqual(auxiliary['incidence'].shape, (2, 16, 5, 3))
        self.assertTrue(torch.allclose(
            auxiliary['incidence'].sum(dim=2),
            torch.ones(2, 16, 3),
            atol=1e-6,
        ))


if __name__ == '__main__':
    unittest.main()
