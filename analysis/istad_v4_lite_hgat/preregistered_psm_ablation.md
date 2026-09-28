# PSM HGAT-Lite paired ablation (frozen before confirmation runs)

Frozen: 2026-09-10, after the exploratory seed-87 pilot and before seeds 90/98.

## Arms

- `legacy`: Causal Conv + legacy HGAT + KAN-TCN + KANAD.
- `lite`: Causal Conv + HGAT-Lite(rank=16) + KAN-TCN + KANAD.
- `none`: Causal Conv + KAN-TCN + KANAD.

All other training, synthetic-corruption, evidence, dual-view, loader, and
evaluation settings are identical.  Confirmation seeds are 90 and 98.  Seed 87
is reported as an exploratory pilot, not silently pooled as confirmation.

## Primary metrics and gate

The HGAT decision uses the neural `model_score`, not the 0.001 innovation-fused
score.  Primary metrics are AUC-PR and AUC-ROC without point adjustment.
Train-p99 raw F1 is a deployment guard; Best-F1+PA is secondary.

Promote HGAT-Lite only if, over the fixed paired runs:

1. mean AP improves over legacy by at least 0.005;
2. mean ROC does not decrease;
3. mean train-p99 raw F1 decreases by no more than 0.010;
4. at least two of three seeds agree in the AP direction;
5. HGAT-Lite also improves over `none`, otherwise remove the spatial branch.

No rank, top-k, hyperedge count, threshold, or fusion weight will be changed in
response to seeds 90/98.

## Frozen-run outcome

Seeds 90/98 completed without configuration changes.  HGAT-Lite improved mean
neural PR-AUC by 0.0473 and Best-F1+PA by 0.0071 versus legacy HGAT, but reduced
ROC-AUC by 0.0084 and train-p99 raw F1 by 0.0624.  The no-spatial arm also beat
HGAT-Lite on mean PR-AUC, ROC-AUC, and Best-F1+PA.  Gates 2, 3, and 5 therefore
failed: HGAT-Lite is not promoted to the default architecture.  Exact values are
stored in `PSM_confirmation_s90_s98.json`.
