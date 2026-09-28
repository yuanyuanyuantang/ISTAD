import argparse
import random
import numpy as np
import torch
import torch.backends

from utils.print_args import print_args


def _resolve_istad_branch_mode(branch_mode, tcn_type):
    if branch_mode:
        return str(branch_mode).lower()
    tcn_type = str(tcn_type).lower()
    return "hgat_tcn" if tcn_type == "standard" else "hgat_kan_tcn"


def _build_setting(args, run_idx):
    arch_tag = '' if getattr(args, 'istad_arch', 'legacy') == 'legacy' else f"_arch{args.istad_arch}"
    abl_tag = '' if getattr(args, 'istad_ablation', 'full') == 'full' else f"_abl{args.istad_ablation}"
    v3_tag = '_v3' if bool(int(getattr(args, 'istad_denoise', 0) or 0)) else ''
    entity_tag = '_ent' if bool(int(getattr(args, 'istad_entity_aware', 0) or 0)) else ''
    entity_context_tag = '_ectx' if bool(int(
        getattr(args, 'istad_entity_context', 0) or 0
    )) else ''
    score_mode = str(getattr(args, 'istad_score_mode', 'base_mean')).lower()
    innovation_tag = '_innov' if score_mode in {
        'innovation', 'innovation_fused', 'innovation_hgat'
    } else ''
    spatial_tag = '_hglite' if str(
        getattr(args, 'istad_spatial_type', 'legacy')
    ).lower() == 'lite' else ''
    relation_input_tag = '_hginnov' if str(
        getattr(args, 'istad_hgat_input', 'raw')
    ).lower() == 'innovation' else ''
    objective = str(getattr(args, 'istad_objective', 'reconstruct')).lower()
    objective_tag = {
        'reconstruct': '',
        'target_reconstruct': '_trecon',
        'target_forecast': '_tfcast',
    }[objective]
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}"
        f"_ft{args.features}_sl{args.seq_len}"
        f"_bm{args.istad_branch_mode}"
        f"{arch_tag}{abl_tag}{v3_tag}{entity_tag}{entity_context_tag}"
        f"{innovation_tag}{spatial_tag}{relation_input_tag}{objective_tag}"
        f"_bs{args.batch_size}_{args.des}_{run_idx}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TSLib Anomaly Detection (ISTAD)')

    # basic
    parser.add_argument('--task_name', type=str, default='anomaly_detection', choices=['anomaly_detection'],
                        help='task name (anomaly-only branch)')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='1=train+test, 0=test only')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='experiment id')
    parser.add_argument('--model', type=str, required=True, choices=['ISTAD'],
                        help='model name')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--setting', type=str, default='',
                        help='optional explicit setting name (mainly for --is_training 0)')

    # data
    parser.add_argument('--data', type=str, required=True, choices=['PSM', 'MSL', 'SMAP', 'SMD', 'SWAT', 'EXATHLON'],
                        help='anomaly dataset name')
    parser.add_argument('--root_path', type=str, default='./dataset/PSM', help='dataset root')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoint directory')
    parser.add_argument('--features', type=str, default='M', choices=['M', 'S', 'MS'], help='feature mode')
    parser.add_argument('--seq_len', type=int, default=96, help='window size')
    parser.add_argument('--enc_in', type=int, default=7, help='input channel size')
    parser.add_argument('--c_out', type=int, default=7, help='output channel size')
    parser.add_argument('--istad_eval_step', type=int, default=0,
                        help='evaluation/scoring window stride; <=0 keeps the legacy loader default')

    # ISTAD model args
    parser.add_argument('--istad_kernel_size', type=int, default=7, help='ISTAD conv kernel size')
    parser.add_argument('--istad_feat_gat_embed_dim', type=int, default=-1,
                        help='ISTAD hypergraph embed dim; <=0 means auto')
    parser.add_argument('--istad_gru_n_layers', type=int, default=1, help='ISTAD GRU layer count')
    parser.add_argument('--istad_gru_hid_dim', type=int, default=150, help='ISTAD GRU hidden dim')
    parser.add_argument('--istad_recon_n_layers', type=int, default=1, help='ISTAD recon decoder layer count')
    parser.add_argument('--istad_recon_hid_dim', type=int, default=150, help='ISTAD recon decoder hidden dim')
    parser.add_argument('--istad_recon_type', type=str, default='kanad',
                        choices=['gru', 'kanad', 'pointwise'],
                        help='ISTAD reconstruction head type')
    parser.add_argument('--istad_kanad_order', type=int, default=3,
                        help='periodic cosine order for KANAD reconstruction head')
    parser.add_argument('--istad_dropout', type=float, default=0.3, help='ISTAD dropout')
    parser.add_argument('--istad_objective', type=str, default='reconstruct',
                        choices=['reconstruct', 'target_reconstruct', 'target_forecast'],
                        help='reconstruct all outputs, selected targets, or causally forecast targets')
    parser.add_argument('--istad_target_features', type=str, default='',
                        help='comma-separated input feature indices predicted by target_forecast')
    parser.add_argument('--istad_forecast_lag', type=int, default=1,
                        help='target-history lag used by target_forecast')
    parser.add_argument('--istad_corrupt_target_only', type=int, default=0, choices=[0, 1],
                        help='restrict synthetic corruptions to target features')
    parser.add_argument('--istad_alpha', type=float, default=0.2, help='ISTAD LeakyReLU alpha')
    parser.add_argument('--istad_n_hyperedges', type=int, default=-1,
                        help='ISTAD hyperedge count; <=0 means auto')
    parser.add_argument('--istad_k_top', type=int, default=-1,
                        help='ISTAD top-k nodes per hyperedge; <=0 means auto')
    parser.add_argument('--istad_h_param_init', type=str, default='normal',
                        choices=['normal', 'xavier', 'uniform'],
                        help='ISTAD H parameter initialization')
    parser.add_argument('--istad_spatial_type', type=str, default='legacy',
                        choices=['legacy', 'lite'],
                        help='legacy dense HGAT or low-rank dynamic hypergraph mixer')
    parser.add_argument('--istad_hgat_rank', type=int, default=16,
                        help='relation rank used by --istad_spatial_type lite')
    parser.add_argument('--istad_hgat_projection', type=str, default='linear',
                        choices=['linear', 'kan', 'brkan'],
                        help='linear, shared KAN, or bounded-residual KAN projection')
    parser.add_argument('--istad_hgat_input', type=str, default='raw',
                        choices=['raw', 'innovation'],
                        help='HGAT input domain: raw normalized values or train-fitted signed VAR innovations')
    parser.add_argument('--istad_branch_mode', type=str, default=None,
                        choices=['hgat', 'tcn', 'kan_tcn', 'hgat_tcn', 'hgat_kan_tcn'],
                        help='ISTAD ablation branch mode; omitted means full hgat + istad_tcn_type branch')
    parser.add_argument('--istad_tcn_type', type=str, default='kan', choices=['standard', 'kan'],
                        help='ISTAD temporal branch type for backward-compatible full-branch mode')
    parser.add_argument('--istad_kan_grid_size', type=int, default=5, help='ISTAD spline grid size')
    parser.add_argument('--istad_kan_spline_order', type=int, default=3, help='ISTAD spline order')
    parser.add_argument('--istad_revin', type=int, default=0,
                        help='enable RevIN (per-window per-feature normalize -> reconstruct -> denormalize)')
    parser.add_argument('--istad_revin_eps', type=float, default=1e-5, help='RevIN variance epsilon')
    parser.add_argument('--istad_dual', type=int, default=0,
                        help='dual-head joint training: shared backbone, base + RevIN reconstruction losses')
    parser.add_argument('--istad_dual_lambda', type=float, default=1.0,
                        help='[dual] weight of RevIN-head reconstruction loss')
    parser.add_argument('--istad_denoise', type=int, default=0,
                        help='ISTAD-v3 clean+synthetic-corruption denoising training (1/0)')
    parser.add_argument('--istad_clean_lambda', type=float, default=1.0,
                        help='[v3] weight of clean reconstruction loss')
    parser.add_argument('--istad_denoise_lambda', type=float, default=1.0,
                        help='[v3] weight of corrupted-input to clean-target reconstruction loss')
    parser.add_argument('--istad_frequency_lambda', type=float, default=0.0,
                        help='[v3] log-spectrum consistency weight inside denoising loss')
    parser.add_argument('--istad_evidence_head', type=int, default=0,
                        help='[v3] train a monotone per-feature anomaly evidence fusion head (1/0)')
    parser.add_argument('--istad_evidence_lambda', type=float, default=0.2,
                        help='[v3] weight of balanced synthetic localization loss')
    parser.add_argument('--istad_point_evidence_lambda', type=float, default=0.0,
                        help='[v3] weight of synthetic point-level pooling supervision')
    parser.add_argument('--istad_corrupt_prob', type=float, default=0.8,
                        help='[v3] probability that a training window receives a synthetic anomaly')
    parser.add_argument('--istad_corrupt_time_ratio', type=float, default=0.15,
                        help='[v3] maximum corrupted segment ratio per selected window')
    parser.add_argument('--istad_corrupt_channel_ratio', type=float, default=0.2,
                        help='[v3] maximum corrupted channel ratio per selected window')
    parser.add_argument('--istad_corrupt_scale', type=float, default=1.5,
                        help='[v3] corruption magnitude relative to local feature scale')
    parser.add_argument('--istad_corrupt_modes', type=int, default=4, choices=[4, 6],
                        help='[v3] number of synthetic corruption families; 4 is the validated default')
    parser.add_argument('--istad_score_mode', type=str, default='base_mean',
                        choices=['base_mean', 'evidence', 'innovation', 'innovation_fused',
                                 'innovation_hgat', 'v7_fused'],
                        help='test score: reconstruction, learned evidence, or the train-only innovation branch')
    parser.add_argument('--istad_score_aggregate', type=str, default='calibrated',
                        choices=['learned', 'topk', 'mean', 'max', 'hybrid',
                                 'calibrated', 'calibrated_mean'],
                        help='[v3] aggregate per-feature evidence into a point score')
    parser.add_argument('--istad_evidence_transform', type=str, default='prob',
                        choices=['prob', 'logit'],
                        help='[v3] map feature logits to bounded probabilities before aggregation')
    parser.add_argument('--istad_score_topk', type=int, default=3,
                        help='[v3] number of strongest feature logits averaged by topk aggregation')
    parser.add_argument('--istad_innovation_lag', type=int, default=1,
                        help='[v4] autoregressive order of the causal innovation branch')
    parser.add_argument('--istad_innovation_ridge', type=float, default=0.01,
                        help='[v4] ridge penalty for the closed-form innovation branch')
    parser.add_argument('--istad_innovation_pool', type=str, default='auto',
                        choices=['auto', 'dense', 'sparse'],
                        help='[v4] innovation pooling; auto uses only normal-train residual statistics')
    parser.add_argument('--istad_innovation_degenerate_cutoff', type=float, default=0.10,
                        help='[v4] auto-pool dense fallback threshold for constant residual channels')
    parser.add_argument('--istad_innovation_scale_floor', type=float, default=0.10,
                        help='[v4] residual-scale floor relative to the median positive scale')
    parser.add_argument('--istad_innovation_fusion_weight', type=float, default=0.001,
                        help='[v4] learned-evidence tie-break weight in innovation_fused mode')
    parser.add_argument('--istad_fusion_recalibrate', type=int, default=0, choices=[0, 1],
                        help='[v4] recalibrate the fused score with its normal-train ECDF (1/0)')
    parser.add_argument('--istad_hgat_fusion_max_weight', type=float, default=0.20,
                        help='[V4-HG] maximum train-gated HGAT contribution')
    parser.add_argument('--istad_hgat_fusion_strategy', type=str, default='rank_tiebreak',
                        choices=['rank_tiebreak', 'reliability_mix'],
                        help='[V4-HG] rank-safe refinement or experimental convex mixing')
    parser.add_argument('--istad_hgat_reliability_reference_fraction', type=float, default=0.80,
                        help='[V4-HG] early normal-train fraction used by the reliability check')
    parser.add_argument('--istad_hgat_reliability_tail_probability', type=float, default=0.01,
                        help='[V4-HG] normal-tail probability checked by the HGAT kill-switch')
    parser.add_argument('--istad_hgat_reliability_inflation_limit', type=float, default=1.25,
                        help='[V4-HG] tolerated HGAT normal-tail inflation relative to V4')
    parser.add_argument('--istad_dump_scores', type=int, default=0,
                        help='save compressed point scores and labels for metric reproduction (1/0)')
    parser.add_argument('--istad_holdout_val', type=int, default=0,
                        help='use disjoint first-80%% train / last-20%% validation split (1/0)')
    parser.add_argument('--istad_entity_aware', type=int, default=0,
                        help='respect known entity boundaries when splitting/windowing (MSL/SMAP/SMD: 1/0)')
    parser.add_argument('--istad_entity_context', type=int, default=0, choices=[0, 1],
                        help='append one-hot entity context nodes after scaling (entity-aware data only)')
    parser.add_argument('--istad_no_test_during_train', type=int, default=0,
                        help='do not inspect test reconstruction loss after every epoch (1/0)')
    parser.add_argument('--istad_arch', type=str, default='legacy',
                        choices=['legacy', 'hgst2', 'v7', 'v71'],
                        help='ISTAD architecture: legacy, hgst2, v7, or residual-gated v71')
    parser.add_argument('--istad_relation_dim', type=int, default=32,
                        help='[hgst2] HGAT relation state dim d_s')
    parser.add_argument('--istad_spatial_grid_size', type=int, default=5,
                        help='[hgst2] spatial spline incidence grid size')
    parser.add_argument('--istad_spatial_spline_order', type=int, default=3,
                        help='[hgst2] spatial spline order')
    parser.add_argument('--istad_hypergraph_temperature', type=float, default=1.0,
                        help='[hgst2] incidence softmax temperature tau_H')
    parser.add_argument('--istad_temporal_width', type=int, default=2,
                        help='[hgst2] per-variable temporal hidden channels r_t')
    parser.add_argument('--istad_temporal_kernel_size', type=int, default=3,
                        help='[hgst2] grouped causal conv kernel size')
    parser.add_argument('--istad_temporal_grid_size', type=int, default=5,
                        help='[hgst2] conditional temporal spline grid size')
    parser.add_argument('--istad_temporal_spline_order', type=int, default=3,
                        help='[hgst2] conditional temporal spline order')
    parser.add_argument('--istad_modulation_rank', type=int, default=2,
                        help='[hgst2] low-rank dim Q of relation-conditioned coefficient modulation')
    parser.add_argument('--istad_modulation_scale', type=float, default=0.05,
                        help='[hgst2] fixed modulation scale eta (0 disables conditioning)')
    parser.add_argument('--istad_decoder_hidden', type=int, default=-1,
                        help='[hgst2] MLP decoder hidden dim; <=0 means N')
    parser.add_argument('--istad_ablation', type=str, default='full',
                        choices=['full', 'fixed_temporal_spline', 'silu_temporal',
                                 'linear_incidence', 'no_spatial'],
                        help='[hgst2] ablation switch (experimental control only)')
    parser.add_argument('--istad_v7_relation_dim', type=int, default=16,
                        help='[v7] low-rank node/target relation dimension')
    parser.add_argument('--istad_v7_use_hypergraph', type=int, default=1, choices=[0, 1],
                        help='[v7] enable directed hypergraph path; 0 is temporal-only control')
    parser.add_argument('--istad_v7_prior_ridge', type=float, default=0.01,
                        help='[v7] training-only signed ridge prior penalty')
    parser.add_argument('--istad_v7_prior_topk', type=int, default=-1,
                        help='[v7] sources retained per target hyperedge; <=0 means auto')
    parser.add_argument('--istad_v7_prior_strength', type=float, default=1.0,
                        help='[v7] log-prior bias strength; 0 is the no-prior control')
    parser.add_argument('--istad_v7_prior_floor', type=float, default=0.01,
                        help='[v7] uniform prior mass allowing dynamic edge discovery')
    parser.add_argument('--istad_v7_prior_lambda', type=float, default=0.05,
                        help='[v7] KL regularization weight toward normal causal prior')
    parser.add_argument('--istad_v7_relation_score_weight', type=float, default=0.25,
                        help='[v7] weight of calibrated relation-deviation score')
    parser.add_argument('--istad_v7_entity_calibration', type=int, default=1, choices=[0, 1],
                        help='[v7] calibrate components per entity using normal training scores')
    parser.add_argument('--istad_v71_graph_gate_init', type=float, default=0.05,
                        help='[v71] initial learned graph-residual gate in (0, 1)')
    parser.add_argument('--istad_v71_prior_gate_init', type=float, default=0.5,
                        help='[v71] initial learned causal-prior mixture gate in (0, 1)')
    parser.add_argument('--istad_enable_explain', type=int, default=1,
                        help='whether to run ISTAD explainability after test (1/0)')
    parser.add_argument('--istad_explain_max_batches', type=int, default=0,
                        help='max test batches for ISTAD explainability; <=0 means all')
    parser.add_argument('--istad_explain_max_points', type=int, default=20000,
                        help='max timeline points in ISTAD saliency heatmap')
    parser.add_argument('--istad_explain_topk', type=int, default=10,
                        help='top-k feature count in ISTAD explainability summary')
    parser.add_argument('--istad_use_smoothgrad', type=int, default=0,
                        help='use SmoothGrad for ISTAD saliency (1/0)')
    parser.add_argument('--istad_smoothgrad_samples', type=int, default=20,
                        help='SmoothGrad samples when enabled')
    parser.add_argument('--istad_smoothgrad_noise', type=float, default=0.1,
                        help='SmoothGrad noise level relative to input std')
    parser.add_argument('--istad_quadrant_threshold_mode', type=str, default='quantile',
                        choices=['quantile', 'median', 'topk', 'fixed'],
                        help='threshold mode for ISTAD quadrant analysis')
    parser.add_argument('--istad_quadrant_threshold_value', type=float, default=0.75,
                        help='threshold value for ISTAD quadrant analysis')
    parser.add_argument('--istad_quadrant_agg_error', type=str, default='sum',
                        choices=['sum', 'mean', 'peak'],
                        help='error aggregation mode in ISTAD segment analysis')
    parser.add_argument('--istad_quadrant_agg_saliency', type=str, default='sum',
                        choices=['sum', 'mean', 'peak'],
                        help='saliency aggregation mode in ISTAD segment analysis')

    # optimization
    parser.add_argument('--anomaly_ratio', type=float, default=1.0, help='anomaly ratio (percent) for threshold')
    parser.add_argument('--istad_threshold_source', type=str, default='combined',
                        choices=['combined', 'train'],
                        help='percentile threshold pool; train avoids using the test score distribution')
    parser.add_argument('--use_bestf1_threshold', type=int, default=1, 
                        help='use Best F1 search for threshold (1=yes, 0=no, requires labels)')
    parser.add_argument('--bestf1_search_mode', type=str, default='adaptive', choices=['manual', 'adaptive'],
                        help='Best F1 search mode: manual (specify range) or adaptive (auto range)')
    parser.add_argument('--bestf1_search_start', type=float, default=0.01,
                        help='Best F1 search start value (manual mode only)')
    parser.add_argument('--bestf1_search_end', type=float, default=2.0,
                        help='Best F1 search end value (manual mode only)')
    parser.add_argument('--bestf1_search_step_num', type=int, default=100,
                        help='Best F1 search step number (manual mode only)')
    parser.add_argument('--bestf1_coarse_step_num', type=int, default=50,
                        help='Best F1 coarse search step number (adaptive mode only)')
    parser.add_argument('--bestf1_fine_step_num', type=int, default=100,
                        help='Best F1 fine search step number (adaptive mode only)')
    parser.add_argument('--bestf1_use_adjustment', type=int, default=1,
                        help='use point adjustment in Best F1 search (1=yes, 0=no)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader num_workers')
    parser.add_argument('--itr', type=int, default=1, help='number of repeated runs')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lradj', type=str, default='type1', choices=['type1', 'type2', 'type3', 'cosine'],
                        help='learning rate adjustment policy')
    parser.add_argument('--des', type=str, default='test', help='experiment description')

    # device
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--gpu_type', type=str, default='cuda', choices=['cuda', 'mps'], help='gpu backend')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='multi-gpu device ids')

    args = parser.parse_args()
    args.istad_branch_mode = _resolve_istad_branch_mode(args.istad_branch_mode, args.istad_tcn_type)
    if args.istad_score_mode in {'evidence', 'innovation_fused'} and not bool(args.istad_evidence_head):
        parser.error(
            '--istad_score_mode evidence/innovation_fused requires --istad_evidence_head 1'
        )
    if not 0.0 <= args.istad_innovation_fusion_weight <= 1.0:
        parser.error('--istad_innovation_fusion_weight must lie in [0, 1]')
    if args.istad_fusion_recalibrate and args.istad_score_mode != 'innovation_fused':
        parser.error('--istad_fusion_recalibrate 1 requires --istad_score_mode innovation_fused')
    if not 0.0 <= args.istad_hgat_fusion_max_weight <= 1.0:
        parser.error('--istad_hgat_fusion_max_weight must lie in [0, 1]')
    if not 0.0 < args.istad_hgat_reliability_reference_fraction < 1.0:
        parser.error('--istad_hgat_reliability_reference_fraction must lie in (0, 1)')
    if not 0.0 < args.istad_hgat_reliability_tail_probability < 0.5:
        parser.error('--istad_hgat_reliability_tail_probability must lie in (0, 0.5)')
    if args.istad_hgat_reliability_inflation_limit < 1.0:
        parser.error('--istad_hgat_reliability_inflation_limit must be at least 1')
    if args.istad_score_mode == 'innovation_hgat':
        if args.istad_arch != 'legacy':
            parser.error('--istad_score_mode innovation_hgat requires --istad_arch legacy')
        if args.istad_spatial_type != 'lite' or 'hgat' not in args.istad_branch_mode:
            parser.error(
                '--istad_score_mode innovation_hgat requires the lightweight HGAT branch'
            )
        if args.istad_dual or args.istad_revin:
            parser.error('innovation_hgat requires a single unnormalized model branch')
    if args.istad_hgat_input == 'innovation' and args.istad_score_mode != 'innovation_hgat':
        parser.error('--istad_hgat_input innovation requires --istad_score_mode innovation_hgat')
    if args.is_training and args.istad_evidence_head and not args.istad_denoise:
        parser.error('training --istad_evidence_head 1 requires --istad_denoise 1')
    if args.istad_entity_aware and args.data not in {'MSL', 'SMAP', 'SMD'}:
        parser.error(
            '--istad_entity_aware 1 currently supports only --data MSL, SMAP, or SMD'
        )
    if args.istad_entity_aware and not args.istad_holdout_val:
        parser.error('--istad_entity_aware 1 requires --istad_holdout_val 1')
    if args.istad_entity_context and not args.istad_entity_aware:
        parser.error('--istad_entity_context 1 requires --istad_entity_aware 1')
    if args.istad_objective in {'target_reconstruct', 'target_forecast'}:
        try:
            target_features = [
                int(value.strip()) for value in args.istad_target_features.split(',')
                if value.strip()
            ]
        except ValueError:
            parser.error('--istad_target_features must be comma-separated integers')
        if not target_features:
            parser.error('--istad_objective target_forecast requires --istad_target_features')
        if len(set(target_features)) != len(target_features):
            parser.error('--istad_target_features cannot contain duplicates')
        if any(index < 0 or index >= args.enc_in for index in target_features):
            parser.error('--istad_target_features contains an index outside [0, enc_in)')
        if len(target_features) != args.c_out:
            parser.error('target feature count must equal --c_out')
        if args.istad_dual:
            parser.error('target-specific ISTAD objectives do not support --istad_dual')
    if args.istad_objective == 'target_forecast':
        if args.istad_forecast_lag <= 0 or args.istad_forecast_lag >= args.seq_len:
            parser.error('--istad_forecast_lag must lie in [1, seq_len)')
    if args.istad_arch in {'v7', 'v71'}:
        if args.istad_objective != 'target_forecast':
            parser.error('--istad_arch v7/v71 requires --istad_objective target_forecast')
        if args.istad_dual or args.istad_revin or args.istad_evidence_head:
            parser.error('ISTAD V7/V7.1 does not support dual, RevIN, or the legacy evidence head')
        if args.istad_denoise:
            parser.error('ISTAD V7/V7.1 uses causal-prior regularization; legacy denoising is disabled')
    if args.istad_score_mode == 'v7_fused' and args.istad_arch not in {'v7', 'v71'}:
        parser.error('--istad_score_mode v7_fused requires --istad_arch v7 or v71')
    if args.istad_v7_relation_dim <= 0:
        parser.error('--istad_v7_relation_dim must be positive')
    if args.istad_v7_prior_ridge <= 0.0:
        parser.error('--istad_v7_prior_ridge must be positive')
    if args.istad_v7_prior_strength < 0.0:
        parser.error('--istad_v7_prior_strength must be non-negative')
    if not 0.0 < args.istad_v7_prior_floor < 1.0:
        parser.error('--istad_v7_prior_floor must lie in (0, 1)')
    if args.istad_v7_prior_lambda < 0.0:
        parser.error('--istad_v7_prior_lambda must be non-negative')
    if not 0.0 <= args.istad_v7_relation_score_weight <= 1.0:
        parser.error('--istad_v7_relation_score_weight must lie in [0, 1]')
    if not 0.0 < args.istad_v71_graph_gate_init < 1.0:
        parser.error('--istad_v71_graph_gate_init must lie in (0, 1)')
    if not 0.0 < args.istad_v71_prior_gate_init < 1.0:
        parser.error('--istad_v71_prior_gate_init must lie in (0, 1)')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
        print('Using GPU')
    else:
        if hasattr(torch.backends, 'mps'):
            args.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        else:
            args.device = torch.device('cpu')
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name != 'anomaly_detection':
        raise ValueError("This branch only supports --task_name anomaly_detection.")

    from exp.exp_anomaly_detection import Exp_Anomaly_Detection
    Exp = Exp_Anomaly_Detection

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = _build_setting(args, ii)
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
    else:
        exp = Exp(args)
        ii = 0
        setting = args.setting if args.setting else _build_setting(args, ii)

        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
