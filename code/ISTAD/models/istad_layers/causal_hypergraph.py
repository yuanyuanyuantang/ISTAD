"""Strictly causal, prior-guided directed hypergraph forecaster for ISTAD V7."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tcn import TemporalConvNet


class CausalPriorHypergraphForecaster(nn.Module):
    """Forecast selected targets with a signed directed causal hypergraph.

    Each target is represented by one directed hyperedge.  Normal-training
    ridge coefficients provide source incidence and signs; a low-rank dynamic
    term adapts incidence at each time point.  All non-convolutional operations
    are pointwise in time and the temporal branch is a causal TCN.
    """

    def __init__(
        self,
        n_features,
        window_size,
        target_indices,
        relation_dim=16,
        prior_topk=None,
        prior_strength=1.0,
        prior_floor=0.01,
        use_hypergraph=True,
        decoder_hidden=32,
        dropout=0.1,
        residual_fusion=False,
        graph_gate_init=0.05,
        prior_gate_init=0.5,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.target_indices = tuple(int(index) for index in target_indices)
        self.n_targets = len(self.target_indices)
        self.relation_dim = int(relation_dim)
        self.prior_strength = float(prior_strength)
        self.prior_floor = float(prior_floor)
        self.use_hypergraph = bool(use_hypergraph)
        self.residual_fusion = bool(residual_fusion)
        self.prior_topk = (
            max(3, int(math.ceil(0.2 * self.n_features)))
            if prior_topk is None or int(prior_topk) <= 0
            else int(prior_topk)
        )
        self.prior_topk = max(1, min(self.prior_topk, self.n_features))
        if self.n_features < 1 or self.n_targets < 1:
            raise ValueError("n_features and number of targets must be positive")
        if any(index < 0 or index >= self.n_features for index in self.target_indices):
            raise ValueError("target index outside feature range")
        if self.relation_dim < 1:
            raise ValueError("relation_dim must be positive")
        if self.prior_strength < 0.0:
            raise ValueError("prior_strength must be non-negative")
        if not 0.0 < self.prior_floor < 1.0:
            raise ValueError("prior_floor must lie in (0, 1)")
        if not 0.0 < float(graph_gate_init) < 1.0:
            raise ValueError("graph_gate_init must lie in (0, 1)")
        if not 0.0 < float(prior_gate_init) < 1.0:
            raise ValueError("prior_gate_init must lie in (0, 1)")

        initial_prior = torch.full(
            (self.n_features, self.n_targets), 1.0 / self.n_features
        )
        initial_sign = torch.ones(self.n_features, self.n_targets)
        self.register_buffer("causal_prior", initial_prior)
        self.register_buffer("causal_sign", initial_sign)
        self.register_buffer("causal_prior_ready", torch.tensor(False))

        self.descriptor = nn.Linear(2, self.relation_dim)
        self.source_embedding = nn.Embedding(self.n_features, self.relation_dim)
        self.target_query = nn.Parameter(
            torch.empty(self.n_targets, self.relation_dim)
        )
        self.state_norm = nn.LayerNorm(self.relation_dim)
        self.edge_norm = nn.LayerNorm(self.relation_dim)
        self.relation_readout = nn.Linear(self.relation_dim, 1)

        tcn_channels = [32, 32, 32, self.n_features]
        self.temporal = TemporalConvNet(
            num_inputs=self.n_features,
            num_channels=tcn_channels,
            kernel_size=9,
            dropout=dropout,
        )
        hidden = max(1, int(decoder_hidden))
        if self.residual_fusion:
            self.temporal_decoder_in = nn.Linear(2, hidden)
            self.temporal_decoder_embedding = nn.Parameter(
                torch.empty(self.n_targets, hidden)
            )
            self.temporal_decoder_norm = nn.LayerNorm(hidden)
            self.temporal_decoder_out = nn.Linear(hidden, 1)
            self.graph_decoder_in = nn.Linear(1, hidden)
            self.graph_decoder_embedding = nn.Parameter(
                torch.empty(self.n_targets, hidden)
            )
            self.graph_decoder_norm = nn.LayerNorm(hidden)
            self.graph_decoder_out = nn.Linear(hidden, 1)
            self.graph_gate_logit = nn.Parameter(torch.full(
                (self.n_targets,), math.log(
                    float(graph_gate_init) / (1.0 - float(graph_gate_init))
                )
            ))
            self.prior_gate_logit = nn.Parameter(torch.full(
                (self.n_targets,), math.log(
                    float(prior_gate_init) / (1.0 - float(prior_gate_init))
                )
            ))
        else:
            self.decoder_in = nn.Linear(3, hidden)
            self.target_decoder_embedding = nn.Parameter(
                torch.empty(self.n_targets, hidden)
            )
            self.decoder_norm = nn.LayerNorm(hidden)
        self.decoder_dropout = nn.Dropout(float(dropout))
        if not self.residual_fusion:
            self.decoder_out = nn.Linear(hidden, 1)

        nn.init.xavier_uniform_(self.descriptor.weight)
        nn.init.zeros_(self.descriptor.bias)
        nn.init.normal_(self.source_embedding.weight, std=0.02)
        nn.init.normal_(self.target_query, std=0.02)
        nn.init.xavier_uniform_(self.relation_readout.weight)
        nn.init.zeros_(self.relation_readout.bias)
        if self.residual_fusion:
            for layer in (
                self.temporal_decoder_in, self.temporal_decoder_out,
                self.graph_decoder_in, self.graph_decoder_out,
            ):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            nn.init.normal_(self.temporal_decoder_embedding, std=0.02)
            nn.init.normal_(self.graph_decoder_embedding, std=0.02)
        else:
            nn.init.xavier_uniform_(self.decoder_in.weight)
            nn.init.zeros_(self.decoder_in.bias)
            nn.init.normal_(self.target_decoder_embedding, std=0.02)
            nn.init.xavier_uniform_(self.decoder_out.weight)
            nn.init.zeros_(self.decoder_out.bias)

    def set_causal_prior(self, prior, signs=None):
        prior = torch.as_tensor(
            prior, dtype=self.causal_prior.dtype, device=self.causal_prior.device
        )
        if tuple(prior.shape) != tuple(self.causal_prior.shape):
            raise ValueError(
                f"expected prior shape {tuple(self.causal_prior.shape)}, "
                f"got {tuple(prior.shape)}"
            )
        if not torch.isfinite(prior).all() or torch.any(prior < 0):
            raise ValueError("causal prior must be finite and non-negative")
        column_sum = prior.sum(dim=0, keepdim=True)
        if torch.any(column_sum <= 0):
            raise ValueError("every target hyperedge needs at least one source")
        prior = prior / column_sum
        self.causal_prior.copy_(prior)

        if signs is not None:
            signs = torch.as_tensor(
                signs, dtype=self.causal_sign.dtype, device=self.causal_sign.device
            )
            if tuple(signs.shape) != tuple(self.causal_sign.shape):
                raise ValueError(
                    f"expected sign shape {tuple(self.causal_sign.shape)}, "
                    f"got {tuple(signs.shape)}"
                )
            if not torch.isfinite(signs).all():
                raise ValueError("causal signs must be finite")
            self.causal_sign.copy_(torch.where(signs < 0, -1.0, 1.0))
        self.causal_prior_ready.fill_(True)

    def _smoothed_prior(self):
        uniform = torch.full_like(self.causal_prior, 1.0 / self.n_features)
        if self.prior_strength == 0.0:
            return uniform
        return (
            (1.0 - self.prior_floor) * self.causal_prior
            + self.prior_floor * uniform
        )

    def _dynamic_incidence(self, state):
        logits = torch.einsum(
            "blnr,cr->blnc", state, self.target_query
        ) / math.sqrt(self.relation_dim)
        prior = self._smoothed_prior()
        if self.prior_strength > 0.0 and not self.residual_fusion:
            logits = logits + self.prior_strength * torch.log(
                prior.clamp_min(1e-12)
            ).view(1, 1, self.n_features, self.n_targets)

        if self.prior_topk < self.n_features:
            top_indices = torch.topk(
                logits, k=self.prior_topk, dim=2
            ).indices
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                2, top_indices, True
            )
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        dynamic = F.softmax(logits, dim=2)
        if self.residual_fusion and self.prior_strength > 0.0:
            prior_gate = torch.sigmoid(self.prior_gate_logit).view(
                1, 1, 1, self.n_targets
            )
            incidence = (
                (1.0 - prior_gate) * dynamic
                + prior_gate * prior.view(1, 1, *prior.shape)
            )
        else:
            incidence = dynamic
        return incidence, prior

    def gate_values(self):
        """Return learned V7.1 gates without exposing mutable parameters."""
        if not self.residual_fusion:
            return None
        graph = torch.sigmoid(self.graph_gate_logit)
        prior = torch.sigmoid(self.prior_gate_logit)
        if not self.use_hypergraph:
            graph = torch.zeros_like(graph)
        if self.prior_strength == 0.0:
            prior = torch.zeros_like(prior)
        return {"graph": graph, "prior": prior}

    @staticmethod
    def _js_deviation(incidence, prior):
        reference = prior.view(1, 1, *prior.shape).expand_as(incidence)
        p = incidence.clamp_min(1e-12)
        q = reference.clamp_min(1e-12)
        middle = 0.5 * (p + q)
        js = 0.5 * (
            torch.sum(p * (torch.log(p) - torch.log(middle)), dim=2)
            + torch.sum(q * (torch.log(q) - torch.log(middle)), dim=2)
        )
        return js.mean(dim=-1)

    def forward(self, x, return_aux=False):
        if x.ndim != 3 or x.shape[1] != self.window_size or x.shape[2] != self.n_features:
            raise ValueError(
                f"expected (B,{self.window_size},{self.n_features}), got {tuple(x.shape)}"
            )

        prior = self._smoothed_prior()
        if self.use_hypergraph:
            delta = torch.zeros_like(x)
            delta[:, 1:] = x[:, 1:] - x[:, :-1]
            descriptor = torch.stack([x, delta], dim=-1)
            state = self.descriptor(descriptor)
            state = self.state_norm(
                state + self.source_embedding.weight.view(
                    1, 1, self.n_features, self.relation_dim
                )
            )
            incidence, prior = self._dynamic_incidence(state)
            edge_state = torch.einsum(
                "blnc,blnr,nc->blcr",
                incidence,
                state,
                self.causal_sign,
            )
            edge_state = self.edge_norm(edge_state)
            relation_value = self.relation_readout(
                F.silu(edge_state)
            ).squeeze(-1)
        else:
            incidence = prior.view(1, 1, *prior.shape).expand(
                x.shape[0], x.shape[1], -1, -1
            )
            relation_value = x.new_zeros(
                x.shape[0], x.shape[1], self.n_targets
            )

        temporal = self.temporal(x.permute(0, 2, 1)).permute(0, 2, 1)
        target_index = torch.as_tensor(self.target_indices, device=x.device)
        temporal_target = temporal.index_select(-1, target_index)
        lagged_target = x.index_select(-1, target_index)
        if self.residual_fusion:
            temporal_features = torch.stack(
                [lagged_target, temporal_target], dim=-1
            )
            temporal_hidden = self.temporal_decoder_in(temporal_features)
            temporal_hidden = temporal_hidden + self.temporal_decoder_embedding.view(
                1, 1, self.n_targets, -1
            )
            temporal_hidden = self.temporal_decoder_norm(temporal_hidden)
            temporal_hidden = self.decoder_dropout(F.silu(temporal_hidden))
            temporal_residual = self.temporal_decoder_out(
                temporal_hidden
            ).squeeze(-1)
            prediction = lagged_target + temporal_residual
            if self.use_hypergraph:
                graph_hidden = self.graph_decoder_in(
                    relation_value.unsqueeze(-1)
                )
                graph_hidden = graph_hidden + self.graph_decoder_embedding.view(
                    1, 1, self.n_targets, -1
                )
                graph_hidden = self.graph_decoder_norm(graph_hidden)
                graph_hidden = self.decoder_dropout(F.silu(graph_hidden))
                graph_residual = self.graph_decoder_out(
                    graph_hidden
                ).squeeze(-1)
                graph_gate = torch.sigmoid(self.graph_gate_logit).view(
                    1, 1, self.n_targets
                )
                prediction = prediction + graph_gate * graph_residual
        else:
            decoder_features = torch.stack(
                [lagged_target, temporal_target, relation_value], dim=-1
            )
            hidden = self.decoder_in(decoder_features)
            hidden = hidden + self.target_decoder_embedding.view(
                1, 1, self.n_targets, -1
            )
            hidden = self.decoder_norm(hidden)
            hidden = self.decoder_dropout(F.silu(hidden))
            prediction = lagged_target + self.decoder_out(hidden).squeeze(-1)

        if not return_aux:
            return prediction

        reference = prior.view(1, 1, *prior.shape).expand_as(incidence)
        if self.use_hypergraph:
            prior_kl = torch.sum(
                incidence.clamp_min(1e-12)
                * (
                    torch.log(incidence.clamp_min(1e-12))
                    - torch.log(reference.clamp_min(1e-12))
                ),
                dim=2,
            ).mean()
        else:
            prior_kl = x.sum() * 0.0
        gates = self.gate_values()
        return prediction, {
            "incidence": incidence,
            "prior_kl": prior_kl,
            "relation_deviation": self._js_deviation(incidence, prior),
            "relation_value": relation_value,
            "graph_gate": None if gates is None else gates["graph"],
            "prior_gate": None if gates is None else gates["prior"],
        }
