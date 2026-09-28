# V4-HG interpretability package: real cases, variable localization, stability

Frozen before the first run on 2026-09-15.  All four benchmark test sets have
already been inspected by earlier versions, so every deliverable below is
development evidence that supports the interpretability narrative; none of it
re-ranks models, selects thresholds, or retrains anything.

## Purpose

Produce the three evidence blocks requested for paper Section 6
("Interpretability Analysis"):

1. **Variable localization metrics** — quantitative, label-free, injection-based.
2. **Real case studies** — qualitative demonstrations on frozen test events.
3. **Stability results** — across injection repetitions and across the three
   frozen HGAT seeds.

## Fixed components (no re-derivation allowed)

- V4 scorer: `code/ISTAD/utils/innovation.py::TrainingOnlyInnovationScorer`
  with lag 1, ridge 1e-2, pool auto, degenerate cutoff 0.10, scale floor 0.10,
  fitted on the flattened strict normal training split exactly as
  `analysis/evaluate_istad_innovation_pa.py` does (loaders with
  `istad_holdout_val=1`, `step=win_size`, SMD entity-aware).
- Per-variable evidence: `TrainingOnlyInnovationScorer.feature_evidence`
  (closed-form, seed-independent, no labels).
- Hyperedge pooling: `utils.innovation.hypergraph_pool_feature_evidence` with
  the dataset's auto-selected pool mode.
- Rank-safe fusion: `s = (q + eps*g)/(1+eps)`, `eps = 0.5/(N_train+1)`.
- Neural forward: `models/ISTAD.Model` in the frozen V4-HG configuration
  (`legacy` arch, `istad_branch_mode hgat`, `istad_spatial_type lite`,
  `istad_hgat_rank 8`, linear projection, raw input; SWAT kernel 15,
  M=20, k=10), weights loaded from
  `checkpoints/ISTAD_v4_hgat_integrated/<frozen setting>/checkpoint.pth`
  (seed 87 for case studies; 87/90/98 for stability).  Checkpoint sha256
  values are cross-checked against the E1 material passport whenever the E1
  per-run JSON is present.
- Data root: `/data/modeluse/TS/ISTAD/dataset`.

## Label-use boundary

Test labels are read only (a) to locate demonstration events for Stage B and
(b) never at all in Stage A.  No threshold, weight, model, or protocol choice
may depend on any test label or test score.  The injection stage operates
exclusively on the label-free normal validation holdout.

## Stage A: injection-based variable localization (label-free)

- Domain: the flattened normal validation holdout of each dataset
  (`flag='val'` loaders; SMD injections confined to a single entity's
  validation tail and never touching an entity's first point).
- N = 200 injections per dataset: 50 per corruption family, families and
  formulas mirrored from the frozen training corruption
  (`exp_anomaly_detection._apply_corruption`, modes 0-3, magnitude 1.5):
  persistent level shift, variance burst, gradual trend (ramp 0.2->1.0),
  local gain (1+0.5u) plus 0.25 offset.
- Amplitude per selected variable: `A_i = 1.5 * u_i * sigma_w`, where
  `sigma_w` is the standard deviation of that variable inside the W-point
  window ending at the segment end (clamped below at 0.1) and
  `u_i = 0.5 + U(0,1)`; sign ± with equal probability.
- Segment length `L ~ U{1..round(0.15 W)}`; injected variables
  `J ~ U{1..round(0.2 C)}`; W and C are the dataset's frozen window and
  channel counts.
- Each injection is applied to a working copy of the validation points,
  scored, then reverted, so injections never interact.
- Evidence ranking: per injected point, rank variables by descending
  e_{t,i} (stable order on ties).
- Metrics, averaged over the segment points and then over injections:
  - Hit@1: the top-ranked variable is injected.
  - Precision@J: fraction of the top-J ranks that are injected (J = |S|).
  - MRR: mean reciprocal rank of the best-ranked injected variable.
- Baselines: (a) per-variable first-difference magnitude |dx_{t,i}| computed
  on the same modified points (no causal model, no training statistics);
  (b) chance level J/C.
- Reported as mean ± bootstrap 95% CI (resampling injections, 1000 draws,
  fixed seed 2026).
- **Sanity gate**: ISTAD Hit@1 CI lower bound must exceed the mean chance
  level J/C.  If it fails, report the failure verbatim; no metric
  redefinition.

## Stage B: real case studies (frozen selection rule)

- Event definition: maximal runs of positive test labels inside one entity
  (SMD) or in the whole series (others), on the flattened tiled test split.
- Selection rule (fixed before running): per dataset take the **longest**
  event; ties broken by earliest start.  Record the rule, the event bounds,
  and the entity id.
- For the selected event, with the seed-87 checkpoint:
  - scores q_t (train ECDF), g_t (graph ECDF on train raw hyperedge
    evidence), s_t (rank-safe fusion) over event ± 2W context;
  - top-5 variables by maximal in-event evidence, with training sigma,
    variable name (SWaT real CSV column names; other datasets use indices);
  - at the in-event evidence peak point: the H_t row of each top-5 variable
    (top-1 hyperedge id and weight) and the ranking of hyperedge evidence
    d_{t,m};
  - one two-panel figure per dataset (score curves + top-variable evidence
    heatmap) and one CSV dump (per-point scores and top-10 variable
    evidence).
- Equivalence anchors (must pass before any output is written):
  1. recomputed q_t equals the archived frozen scores
     `analysis/istad_v4_innovation/<DS>_innovation_scores.npz` (field
     `score`) at identical indices within 1e-6, whenever the archive exists;
  2. the seed-87 checkpoint sha256 matches the E1 material passport when
     present;
  3. train-window graph scores are computed once per dataset and cached;
     a rerun must reproduce them bitwise.

## Stage C: stability

- S1: bootstrap CIs from Stage A (above).
- S2: cross-seed H_t agreement.  For seeds 87/90/98, forward the same
  event windows and compute, per ordered seed pair, (i) the mean Jaccard
  agreement of the variable->top-1-hyperedge assignment over event points
  and (ii) Kendall tau between hyperedge evidence vectors d_{t,m} at the
  event peak point.
- S3: the per-variable evidence branch is closed-form and contains no
  random state; cross-seed identity is asserted by construction and stated
  in the results.
- Boundary statement (mandatory in the write-up): E1 showed the learned
  incidence is metric-invariant under degree-preserving counterfactuals and
  R1's masked-channel gate failed; therefore localization capability is
  attributed to the auditable causal-innovation evidence, and H_t is
  presented as a transparent relational organization of that evidence whose
  score contribution is bounded by the rank-safe constraint.

## Outputs

```
analysis/v4hg_interpretability/
  PREREGISTERED_PROTOCOL.md     (this file, written before any run)
  RESULTS.md                    (tables + verdicts)
  localization_metrics.json     (Stage A)
  case_studies/                 (per-dataset PNG + CSV + case JSON)
  stability.json                (Stage C)
```

Single entry point: `python analysis/evaluate_v4hg_interpretability.py
--stage all --device cuda:0`.  Fixed seed 87 for any random consumption;
rerunning `--stage all` must reproduce JSON/CSV outputs identically.
