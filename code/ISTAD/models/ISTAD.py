import torch
import torch.nn as nn

from .istad_layers import (
    ConvLayer,
    HypergraphAttentionLayer,
    LightweightHypergraphMixer,
    GRULayer,
    TemporalConvNet,
    TemporalKANNet,
    ReconstructionModel,
    KANADReconstructionModel,
    PointwisePredictionModel,
    EvidenceFusionHead,
    ISTADHGST2,
    CausalPriorHypergraphForecaster,
)
from .istad_layers.relation_bottleneck import (
    ChannelwiseRelationPredictionModel,
    DepthwiseCausalEncoder,
)


def _as_optional_int(value):
    if value is None:
        return None
    value = int(value)
    return None if value <= 0 else value


_VALID_BRANCH_MODES = {"hgat", "tcn", "kan_tcn", "hgat_tcn", "hgat_kan_tcn"}


def _resolve_branch_mode(branch_mode, tcn_type="kan"):
    if branch_mode is None or str(branch_mode).strip() == "":
        tcn_type = str(tcn_type).lower()
        return "hgat_tcn" if tcn_type == "standard" else "hgat_kan_tcn"

    branch_mode = str(branch_mode).lower()
    if branch_mode not in _VALID_BRANCH_MODES:
        raise ValueError(
            f"Unsupported branch_mode={branch_mode}. "
            "Use one of: hgat, tcn, kan_tcn, hgat_tcn, hgat_kan_tcn."
        )
    return branch_mode


class ISTAD(nn.Module):
    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        feat_gat_embed_dim=None,
        gru_n_layers=1,
        gru_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.3,
        alpha=0.2,
        n_hyperedges=None,
        k_top=5,
        h_param_init="normal",
        spatial_type="legacy",
        hgat_rank=16,
        hgat_projection="linear",
        hgat_relation_mode="dynamic",
        hgat_relation_seed=0,
        tcn_type="kan",
        branch_mode="hgat_kan_tcn",
        kan_grid_size=5,
        kan_spline_order=3,
        recon_type="gru",
        kanad_order=3,
        relation_bottleneck=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.out_dim = out_dim

        if n_hyperedges is None:
            n_hyperedges = max(1, int(n_features // 2))
        if feat_gat_embed_dim is None:
            feat_gat_embed_dim = 32

        self.relation_bottleneck = bool(relation_bottleneck)
        self.conv = (
            DepthwiseCausalEncoder(n_features, kernel_size)
            if self.relation_bottleneck
            else ConvLayer(n_features, kernel_size)
        )
        self.branch_mode = _resolve_branch_mode(branch_mode, tcn_type)
        self.use_hgat = "hgat" in self.branch_mode
        self.spatial_type = str(spatial_type).lower()
        if self.spatial_type not in {"legacy", "lite"}:
            raise ValueError("spatial_type must be 'legacy' or 'lite'")
        if self.branch_mode in {"tcn", "hgat_tcn"}:
            self.temporal_branch = "standard"
        elif self.branch_mode in {"kan_tcn", "hgat_kan_tcn"}:
            self.temporal_branch = "kan"
        else:
            self.temporal_branch = None

        recon_in_dim = n_features
        if self.use_hgat:
            if self.spatial_type == "lite":
                self.node_project_in = nn.Identity()
                self.feature_gat = LightweightHypergraphMixer(
                    n_nodes=n_features,
                    n_hyperedges=n_hyperedges,
                    rank=hgat_rank,
                    projection_type=hgat_projection,
                    kan_grid_size=kan_grid_size,
                    kan_spline_order=kan_spline_order,
                    dropout=dropout,
                    k_top=k_top,
                    relation_mode=hgat_relation_mode,
                    relation_seed=hgat_relation_seed,
                )
                self.node_project_out = nn.Identity()
            else:
                self.node_project_in = nn.Linear(1, feat_gat_embed_dim)
                self.feature_gat = HypergraphAttentionLayer(
                    in_features=feat_gat_embed_dim,
                    out_features=feat_gat_embed_dim,
                    n_nodes=n_features,
                    n_hyperedges=n_hyperedges,
                    dropout=dropout,
                    alpha=alpha,
                    k_top=k_top,
                    h_param_init=h_param_init,
                )
                self.node_project_out = nn.Linear(feat_gat_embed_dim, 1)
            recon_in_dim += n_features

        if self.temporal_branch is not None:
            tcn_channels = [32, 32, 32, n_features]
            if self.temporal_branch == "kan":
                self.tcn = TemporalKANNet(
                    num_inputs=n_features,
                    num_channels=tcn_channels,
                    kernel_size=9,
                    dropout=dropout,
                    grid_size=kan_grid_size,
                    spline_order=kan_spline_order,
                )
            else:
                self.tcn = TemporalConvNet(
                    num_inputs=n_features,
                    num_channels=tcn_channels,
                    kernel_size=9,
                    dropout=dropout,
                )
            recon_in_dim += n_features

        self.recon_in_dim = recon_in_dim

        self.recon_type = str(recon_type).lower()
        if self.recon_type not in {"gru", "kanad", "pointwise"}:
            raise ValueError(
                f"Unsupported recon_type={recon_type}. Use 'gru', 'kanad', or 'pointwise'."
            )

        if self.relation_bottleneck:
            if not self.use_hgat or self.spatial_type != "lite":
                raise ValueError(
                    "relation_bottleneck requires the lightweight HGAT pathway"
                )
            if self.temporal_branch is not None or self.recon_type != "pointwise":
                raise ValueError(
                    "relation_bottleneck requires branch_mode='hgat' and "
                    "recon_type='pointwise'"
                )
            if out_dim != n_features:
                raise ValueError(
                    "relation_bottleneck requires out_dim == n_features"
                )
            self.gru = None
            self.recon_model = ChannelwiseRelationPredictionModel(
                n_channels=n_features,
                hidden_dim=recon_hid_dim,
                dropout=dropout,
            )
        elif self.recon_type == "gru":
            self.gru = GRULayer(self.recon_in_dim, gru_hid_dim, gru_n_layers, dropout)
            self.recon_model = ReconstructionModel(
                window_size,
                gru_hid_dim,
                recon_hid_dim,
                out_dim,
                recon_n_layers,
                dropout,
            )
        elif self.recon_type == "kanad":
            self.gru = None
            self.recon_model = KANADReconstructionModel(
                window_size=window_size,
                in_dim=self.recon_in_dim,
                out_dim=out_dim,
                order=kanad_order,
            )
        else:
            self.gru = None
            self.recon_model = PointwisePredictionModel(
                in_dim=self.recon_in_dim,
                out_dim=out_dim,
                hidden_dim=recon_hid_dim,
                dropout=dropout,
            )

    def forward(self, x, return_hgat_aux=False):
        conv_output = self.conv(x)
        if conv_output.shape[1] != self.window_size:
            raise ValueError(
                f"Conv branch changed window length: {conv_output.shape[1]} != {self.window_size}"
            )

        batch_size, window_size, n_features = conv_output.shape
        branch_outputs = [conv_output]

        hgat_auxiliary = None
        if self.use_hgat:
            x_spatial = conv_output.unsqueeze(-1)
            x_spatial = self.node_project_in(x_spatial)
            x_spatial = x_spatial.reshape(batch_size * window_size, n_features, -1)
            if return_hgat_aux:
                if self.spatial_type != "lite":
                    raise ValueError(
                        "HGAT innovation scoring requires spatial_type='lite'"
                    )
                h_feat, reverse, incidence = self.feature_gat(
                    x_spatial, return_extras=True
                )
                hgat_auxiliary = {
                    "incidence": incidence.reshape(
                        batch_size, window_size, n_features, -1
                    ),
                    "reverse_incidence": reverse.reshape(
                        batch_size, window_size, n_features, -1
                    ),
                    "message_gate": torch.sigmoid(
                        self.feature_gat.message_gate_logit
                    ),
                }
            else:
                h_feat = self.feature_gat(x_spatial)
            h_feat = self.node_project_out(h_feat)
            h_feat = h_feat.reshape(batch_size, window_size, n_features)
            if hgat_auxiliary is not None:
                hgat_auxiliary["spatial_output"] = h_feat
            branch_outputs.append(h_feat)
        elif return_hgat_aux:
            raise ValueError("HGAT innovation scoring requires an HGAT branch")

        if self.temporal_branch is not None:
            x_temp = conv_output.permute(0, 2, 1)
            x_temp = self.tcn(x_temp)
            if x_temp.shape[2] != self.window_size:
                raise ValueError(
                    f"TCN branch changed window length: {x_temp.shape[2]} != {self.window_size}"
                )
            h_temp = x_temp.permute(0, 2, 1)
            branch_outputs.append(h_temp)

        h_cat = torch.cat(branch_outputs, dim=2)
        if h_cat.size(-1) != self.recon_in_dim:
            raise ValueError(
                f"Reconstruction input mismatch: expected {self.recon_in_dim}, got {h_cat.size(-1)}"
            )
        if self.relation_bottleneck:
            recons = self.recon_model(conv_output, h_feat)
        elif self.recon_type == "gru":
            gru_out_seq, _ = self.gru(h_cat)
            recons = self.recon_model(gru_out_seq)
        else:
            recons = self.recon_model(h_cat)
        if return_hgat_aux:
            return recons, hgat_auxiliary
        return recons


# legacy 并行拼接模型的显式别名（仅命名，不改行为；用于论文消融对照 "Parallel concatenation"）
LegacyParallelISTAD = ISTAD


class Model(nn.Module):
    """TSLib-compatible wrapper of the ISTAD anomaly model.

    istad_arch 分派：
    - 'legacy'（默认）：三路并行拼接模型（上方 ISTAD 类，别名 LegacyParallelISTAD）
    - 'hgst2'：HGST v2 统一时空块 + MLP decoder
    - 'v7'：训练因果先验约束的有向动态超图预测器
    - 'v71'：带可退化图残差与可学习先验门的 V7.1
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.objective = str(getattr(configs, "istad_objective", "reconstruct")).lower()

        arch = str(getattr(configs, "istad_arch", "legacy")).lower()
        self.istad_arch = arch

        # RevIN（Reversible Instance Normalization）：逐窗口按特征去均值/方差，
        # 重建后再反归一化回原空间。使模型对窗口内水平/尺度漂移不变
        # （针对 PSM f3、SWAT f5/f26/f35 之类的 train→test 分布偏移）。
        # 注意：dotdict 对缺失键返回 None（getattr 默认值不生效），需显式兜底
        self.use_revin = bool(int(getattr(configs, "istad_revin", 0) or 0))
        self.revin_eps = float(getattr(configs, "istad_revin_eps", None) or 1e-5)
        # 双头联合训练：同一 backbone 做两次前向（原始空间 + RevIN 空间），
        # 两路重建损失联合优化，输出沿特征维拼接为 (B, sl, 2C)。
        # 动机：独立训练的 RevIN 模型表征退化（PSM ROC 0.55 / SWAT 0.24 反相关），
        # 联合训练用 base 损失约束共享表征，保住 RevIN 视角的排序能力。
        self.use_dual = bool(int(getattr(configs, "istad_dual", 0) or 0))
        if self.use_dual:
            self.use_revin = False  # dual 自带两路，避免单路覆盖

        n_features = int(configs.enc_in)
        out_dim = int(getattr(configs, "c_out", n_features))
        target_text = str(getattr(configs, "istad_target_features", "") or "")
        self.target_indices = tuple(
            int(value.strip()) for value in target_text.split(",") if value.strip()
        )
        if self.objective in {"target_reconstruct", "target_forecast"}:
            if len(self.target_indices) != out_dim:
                raise ValueError("target-specific objective requires one target index per output")
        else:
            self.target_indices = tuple(range(out_dim))
        self.use_evidence_head = bool(int(getattr(configs, "istad_evidence_head", 0) or 0))
        if self.use_evidence_head:
            if self.objective == "reconstruct" and out_dim != n_features:
                raise ValueError("reconstruction evidence head requires c_out == enc_in")
            self.evidence_head = EvidenceFusionHead(out_dim, dual=self.use_dual)

        if arch in {"v7", "v71"}:
            if self.objective != "target_forecast":
                raise ValueError("ISTAD V7 requires objective='target_forecast'")
            prior_topk = _as_optional_int(getattr(configs, "istad_v7_prior_topk", -1))
            self.backbone = CausalPriorHypergraphForecaster(
                n_features=n_features,
                window_size=self.seq_len,
                target_indices=self.target_indices,
                relation_dim=int(getattr(configs, "istad_v7_relation_dim", 16)),
                prior_topk=prior_topk,
                prior_strength=float(getattr(configs, "istad_v7_prior_strength", 1.0)),
                prior_floor=float(getattr(configs, "istad_v7_prior_floor", 0.01)),
                use_hypergraph=bool(int(getattr(
                    configs, "istad_v7_use_hypergraph", 1
                ) or 0)),
                decoder_hidden=int(getattr(configs, "istad_recon_hid_dim", 32)),
                dropout=float(getattr(configs, "istad_dropout", 0.1)),
                residual_fusion=arch == "v71",
                graph_gate_init=float(getattr(
                    configs, "istad_v71_graph_gate_init", 0.05
                )),
                prior_gate_init=float(getattr(
                    configs, "istad_v71_prior_gate_init", 0.5
                )),
            )
            return

        if arch == "hgst2":
            import math
            n_hyperedges_cfg = _as_optional_int(getattr(configs, "istad_n_hyperedges", -1))
            if n_hyperedges_cfg is None:
                n_hyperedges_cfg = max(4, math.ceil(n_features / 2))
            k_top_cfg = _as_optional_int(getattr(configs, "istad_k_top", -1))
            if k_top_cfg is None:
                k_top_cfg = max(3, math.ceil(0.2 * n_features))
            decoder_hidden = _as_optional_int(getattr(configs, "istad_decoder_hidden", -1))

            self.backbone = ISTADHGST2(
                n_features=n_features,
                window_size=self.seq_len,
                out_dim=out_dim,
                relation_dim=int(getattr(configs, "istad_relation_dim", 32)),
                n_hyperedges=n_hyperedges_cfg,
                k_top=k_top_cfg,
                spatial_grid_size=int(getattr(configs, "istad_spatial_grid_size", 5)),
                spatial_spline_order=int(getattr(configs, "istad_spatial_spline_order", 3)),
                hypergraph_temperature=float(getattr(configs, "istad_hypergraph_temperature", 1.0)),
                temporal_width=int(getattr(configs, "istad_temporal_width", 2)),
                temporal_kernel_size=int(getattr(configs, "istad_temporal_kernel_size", 3)),
                temporal_grid_size=int(getattr(configs, "istad_temporal_grid_size", 5)),
                temporal_spline_order=int(getattr(configs, "istad_temporal_spline_order", 3)),
                modulation_rank=int(getattr(configs, "istad_modulation_rank", 2)),
                modulation_scale=float(getattr(configs, "istad_modulation_scale", 0.05)),
                dropout=float(getattr(configs, "istad_dropout", 0.1)),
                decoder_hidden=decoder_hidden,
                ablation=str(getattr(configs, "istad_ablation", "full")),
            )
            return

        if arch != "legacy":
            raise ValueError(
                f"Unsupported istad_arch={arch}. Use 'legacy', 'hgst2', 'v7', or 'v71'."
            )


        kernel_size = int(getattr(configs, "istad_kernel_size", 7))
        feat_gat_embed_dim = _as_optional_int(getattr(configs, "istad_feat_gat_embed_dim", -1))
        gru_n_layers = int(getattr(configs, "istad_gru_n_layers", 1))
        gru_hid_dim = int(getattr(configs, "istad_gru_hid_dim", 150))
        recon_n_layers = int(getattr(configs, "istad_recon_n_layers", 1))
        recon_hid_dim = int(getattr(configs, "istad_recon_hid_dim", 150))
        dropout = float(getattr(configs, "istad_dropout", 0.3))
        alpha = float(getattr(configs, "istad_alpha", 0.2))

        n_hyperedges_cfg = _as_optional_int(getattr(configs, "istad_n_hyperedges", -1))
        if n_hyperedges_cfg is None:
            n_hyperedges = max(1, int(n_features // 2))
        else:
            n_hyperedges = n_hyperedges_cfg

        k_top_cfg = _as_optional_int(getattr(configs, "istad_k_top", -1))
        if k_top_cfg is None:
            k_top = max(1, int(n_features * 0.2))
        else:
            k_top = k_top_cfg

        h_param_init = str(getattr(configs, "istad_h_param_init", "normal"))
        spatial_type = str(getattr(configs, "istad_spatial_type", "legacy"))
        hgat_rank = int(getattr(configs, "istad_hgat_rank", 16))
        hgat_projection = str(
            getattr(configs, "istad_hgat_projection", "linear") or "linear"
        ).lower()
        hgat_relation_mode = str(
            getattr(configs, "istad_hgat_relation_mode", "dynamic") or "dynamic"
        ).lower()
        hgat_relation_seed = int(getattr(configs, "seed", 0))
        tcn_type = str(getattr(configs, "istad_tcn_type", "kan")).lower()
        if tcn_type not in {"standard", "kan"}:
            tcn_type = "kan"
        branch_mode = _resolve_branch_mode(getattr(configs, "istad_branch_mode", None), tcn_type)
        kan_grid_size = int(getattr(configs, "istad_kan_grid_size", 5))
        kan_spline_order = int(getattr(configs, "istad_kan_spline_order", 3))

        recon_type = str(getattr(configs, "istad_recon_type", "gru")).lower()
        if recon_type not in {"gru", "kanad", "pointwise"}:
            recon_type = "gru"
        kanad_order = int(getattr(configs, "istad_kanad_order", 3))
        if kanad_order <= 0:
            kanad_order = 3

        self.backbone = ISTAD(
            n_features=n_features,
            window_size=self.seq_len,
            out_dim=out_dim,
            kernel_size=kernel_size,
            feat_gat_embed_dim=feat_gat_embed_dim,
            gru_n_layers=gru_n_layers,
            gru_hid_dim=gru_hid_dim,
            recon_n_layers=recon_n_layers,
            recon_hid_dim=recon_hid_dim,
            dropout=dropout,
            alpha=alpha,
            n_hyperedges=n_hyperedges,
            k_top=k_top,
            h_param_init=h_param_init,
            spatial_type=spatial_type,
            hgat_rank=hgat_rank,
            hgat_projection=hgat_projection,
            hgat_relation_mode=hgat_relation_mode,
            hgat_relation_seed=hgat_relation_seed,
            tcn_type=tcn_type,
            branch_mode=branch_mode,
            kan_grid_size=kan_grid_size,
            kan_spline_order=kan_spline_order,
            recon_type=recon_type,
            kanad_order=kanad_order,
            relation_bottleneck=bool(int(getattr(
                configs, "istad_relation_bottleneck", 0
            ) or 0)),
        )

    def _revin_forward(self, x_enc):
        # 逐窗口、逐特征：均值/方差在时间维（dim=1）上统计
        mean = x_enc.mean(dim=1, keepdim=True)
        var = x_enc.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.revin_eps)
        x_norm = (x_enc - mean) / std
        recon_norm = self.backbone(x_norm)
        return recon_norm * std + mean

    def anomaly_detection(self, x_enc, return_aux=False):
        if self.istad_arch in {"v7", "v71"}:
            return self.backbone(x_enc, return_aux=return_aux)
        if return_aux:
            if self.use_dual or self.use_revin:
                raise ValueError(
                    "HGAT auxiliary output requires a single unnormalized branch"
                )
            return self.backbone(x_enc, return_hgat_aux=True)
        if self.use_dual:
            base = self.backbone(x_enc)
            revin = self._revin_forward(x_enc)
            # 沿特征维拼接 (B, sl, 2C)；DataParallel gather 兼容，
            # 由 exp 侧按 C 切分（base=前 C 列, revin=后 C 列）。
            return torch.cat([base, revin], dim=-1)
        if self.use_revin:
            return self._revin_forward(x_enc)
        return self.backbone(x_enc)

    def set_causal_prior(self, prior, signs=None):
        if self.istad_arch not in {"v7", "v71"}:
            raise RuntimeError("causal priors are available only for ISTAD V7/V7.1")
        self.backbone.set_causal_prior(prior, signs)

    def evidence_logits(self, x_enc, outputs):
        """Return per-time, per-feature anomaly logits from reconstruction evidence."""
        if not self.use_evidence_head:
            raise RuntimeError("Enable --istad_evidence_head 1 before requesting learned scores")
        if self.objective in {"target_reconstruct", "target_forecast"}:
            index = torch.as_tensor(self.target_indices, device=x_enc.device)
            observed = x_enc.index_select(-1, index)
        else:
            observed = x_enc
        n_features = observed.size(-1)
        if self.use_dual:
            base = outputs[..., :n_features]
            revin = outputs[..., n_features:2 * n_features]
            return self.evidence_head((observed - base).square(), (observed - revin).square())
        return self.evidence_head((observed - outputs[..., :n_features]).square())

    def aggregate_evidence(self, feature_score, mode="learned", topk=3):
        if not self.use_evidence_head:
            raise RuntimeError("Enable --istad_evidence_head 1 before requesting learned scores")
        return self.evidence_head.aggregate(feature_score, mode=mode, topk=topk)

    def forward(
        self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_aux=False
    ):
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc, return_aux=return_aux)
        raise NotImplementedError("ISTAD in this branch supports anomaly_detection only.")
