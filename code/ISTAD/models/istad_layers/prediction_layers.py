"""Reconstruction heads used in ISTAD anomaly detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gru_layer import RNNDecoder


class ReconstructionModel(nn.Module):
    """GRU-based reconstruction head."""

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = int(window_size)
        self.in_dim = int(in_dim)
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (B, W, C), but got shape {tuple(x.shape)}")
        if x.size(1) != self.window_size:
            raise ValueError(f"Window length mismatch: expected {self.window_size}, got {x.size(1)}")
        if x.size(-1) != self.in_dim:
            raise ValueError(f"Feature mismatch: expected {self.in_dim}, got {x.size(-1)}")

        decoder_out = self.decoder(x)  # (B, W, hid_dim)
        return self.fc(decoder_out)  # (B, W, out_dim)


class _KANADCore(nn.Module):
    """KANAD temporal block for a single-channel sequence."""

    def __init__(self, window_size, order):
        super().__init__()
        self.window_size = int(window_size)
        self.order = int(order)
        if self.order <= 0:
            raise ValueError(f"KANAD order must be positive, got {self.order}")

        self.channels = 2 * self.order + 1
        self.register_buffer(
            "orders",
            self._create_custom_periodic_cosine(self.window_size, self.order).unsqueeze(0),
        )  # (1, order, window_size)

        self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(1)
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Linear(self.window_size, self.window_size)

    @staticmethod
    def _create_custom_periodic_cosine(window_size, order):
        periods = [float(i) for i in range(1, int(order) + 1)]
        result = torch.empty(len(periods), int(window_size), dtype=torch.float32)
        base = torch.arange(int(window_size), dtype=torch.float32) / float(window_size)
        for idx, period in enumerate(periods):
            result[idx, :] = torch.cos((base / period) * (2.0 * torch.pi))
        return result

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (N, W), but got shape {tuple(x.shape)}")
        if x.size(1) != self.window_size:
            raise ValueError(f"Window length mismatch: expected {self.window_size}, got {x.size(1)}")

        x_expanded = x.unsqueeze(1)
        residuals = [x_expanded]

        ff = torch.cat(
            [self.orders.repeat(x.size(0), 1, 1)]
            + [torch.cos(k * x_expanded) for k in range(1, self.order + 1)]
            + [x_expanded],
            dim=1,
        )  # (N, 2*order+1, W)
        residuals.append(ff)

        ff = self.init_conv(ff)
        ff = self.bn1(ff)
        ff = self.act(ff)
        ff = self.inner_conv(ff) + residuals.pop()
        ff = self.bn2(ff)
        ff = self.act(ff)
        ff = self.out_conv(ff) + residuals.pop()
        ff = self.bn3(ff)
        ff = self.act(ff)
        ff = self.final_conv(ff)
        return ff.squeeze(1)  # (N, W)


class KANADReconstructionModel(nn.Module):
    """KANAD-style reconstruction head for ISTAD."""

    def __init__(self, window_size, in_dim, out_dim, order):
        super().__init__()
        self.window_size = int(window_size)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.input_proj = nn.Linear(self.in_dim, self.out_dim)
        self.kanad = _KANADCore(self.window_size, order)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (B, W, C), but got shape {tuple(x.shape)}")
        if x.size(1) != self.window_size:
            raise ValueError(f"Window length mismatch: expected {self.window_size}, got {x.size(1)}")
        if x.size(2) != self.in_dim:
            raise ValueError(f"Feature mismatch: expected {self.in_dim}, got {x.size(2)}")

        x_projected = self.input_proj(x)  # (B, W, out_dim)
        batch_size = x_projected.size(0)

        # Apply one shared KANAD block per output channel, matching the original KANAD style.
        x_flat = x_projected.permute(0, 2, 1).contiguous().view(batch_size * self.out_dim, self.window_size)
        recon_flat = self.kanad(x_flat)  # (B*out_dim, W)
        return recon_flat.view(batch_size, self.out_dim, self.window_size).permute(0, 2, 1).contiguous()


class PointwisePredictionModel(nn.Module):
    """Small per-time-step decoder that cannot mix information across time."""

    def __init__(self, in_dim, out_dim, hidden_dim=32, dropout=0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = max(1, int(hidden_dim))
        self.network = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(self, x):
        if x.ndim != 3 or x.size(-1) != self.in_dim:
            raise ValueError(
                f"Expected (B,W,{self.in_dim}) input, got {tuple(x.shape)}"
            )
        return self.network(x)


class EvidenceFusionHead(nn.Module):
    """Sign-constrained, channel-aware fusion of reconstruction evidence.

    The head is deliberately small and constrained: every evidence coefficient is
    positive (softplus parameterisation).  The direct base/RevIN error terms are
    monotone when the remaining evidence is held fixed; the disagreement term is
    intentionally non-monotone in either view alone.  It is trained only with
    synthetic corruptions of normal training windows and therefore does not
    require test labels or test-distribution statistics.
    """

    def __init__(self, n_features, dual=False):
        super().__init__()
        self.n_features = int(n_features)
        self.dual = bool(dual)
        self.n_evidence = 3 if self.dual else 1

        # softplus(0) gives a neutral positive initial weight.  Per-channel bias
        # absorbs the different normal error floors of heterogeneous sensors.
        self.raw_weight = nn.Parameter(torch.zeros(self.n_features, self.n_evidence))
        self.bias = nn.Parameter(torch.full((self.n_features,), -2.0))
        # A single interpretable gate learns whether anomalies in this dataset are
        # better represented by dense multi-sensor evidence or a sparse top-k
        # response.  It is supervised only by synthetic validation/training masks.
        self.pool_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, base_error, revin_error=None):
        if base_error.ndim != 3 or base_error.size(-1) != self.n_features:
            raise ValueError(
                f"Expected base_error (B,W,{self.n_features}), got {tuple(base_error.shape)}"
            )

        base_evidence = torch.log1p(base_error.clamp_min(0.0))
        evidence = [base_evidence]
        if self.dual:
            if revin_error is None or revin_error.shape != base_error.shape:
                raise ValueError("dual EvidenceFusionHead requires matching revin_error")
            revin_evidence = torch.log1p(revin_error.clamp_min(0.0))
            # Cross-view disagreement is a non-negative evidence term useful under
            # distribution shift.  It is not coordinate-wise monotone because
            # moving one view toward the other reduces disagreement.
            disagreement = (base_evidence - revin_evidence).abs()
            evidence.extend([revin_evidence, disagreement])

        stacked = torch.stack(evidence, dim=-1)  # (B,W,C,E)
        weight = F.softplus(self.raw_weight).view(1, 1, self.n_features, self.n_evidence)
        return (stacked * weight).sum(dim=-1) + self.bias.view(1, 1, self.n_features)

    def aggregate(self, feature_score, mode="learned", topk=3):
        """Aggregate feature evidence without consulting real anomaly labels."""
        if feature_score.ndim != 3 or feature_score.size(-1) != self.n_features:
            raise ValueError(
                f"Expected feature_score (B,W,{self.n_features}), got {tuple(feature_score.shape)}"
            )
        mode = str(mode).lower()
        dense_score = feature_score.mean(dim=-1)
        if mode == "mean":
            return dense_score
        if mode == "max":
            return feature_score.max(dim=-1).values

        k = min(self.n_features, max(1, int(topk)))
        sparse_score = feature_score.topk(k, dim=-1).values.mean(dim=-1)
        if mode == "topk":
            return sparse_score
        if mode == "hybrid":
            return 0.5 * (dense_score + sparse_score)
        if mode == "learned":
            sparse_weight = self.pool_logit.sigmoid()
            return (1.0 - sparse_weight) * dense_score + sparse_weight * sparse_score
        raise ValueError(f"Unsupported evidence aggregation: {mode}")
