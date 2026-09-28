# V10 projection ablation protocol

Written after the frozen V10 continuation gate failed and before running this
ablation on 2026-09-11.  This is a post-hoc explanatory experiment on already
inspected test sets; it cannot promote V10 or reopen seeds 90/98.

Run one seed-87 linear-projection arm on EXATHLON, PSM, SMD and SWAT.  It must
match V10 in signed train-standardized VAR-innovation input, causal Conv,
rank-8 HGAT-Lite, normal-only denoising, graph-evidence pooling, training-only
ECDF and reliability mixture.  The only changed factor is `brkan -> linear`.

Retain BR-KAN as a supported mechanism only if its Macro AP exceeds the linear
arm and at least two of four per-dataset AP changes are non-negative.  Otherwise
prefer the smaller linear projection.  Do not tune either arm after evaluation.
