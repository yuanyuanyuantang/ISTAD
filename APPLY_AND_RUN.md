# ISTAD E1 server execution

This patch adds an opt-in, inference-only incidence counterfactual audit. It does not retrain the model and redirects all generated evaluation files away from the frozen `test_results` directory.

## 1. Verify and install

Run from `/data/modeluse/TS/ISTAD-release` before extracting the patch:

```bash
sha256sum \
  code/ISTAD/models/ISTAD.py \
  code/ISTAD/models/istad_layers/hypergraph_attention.py \
  code/ISTAD/exp/exp_anomaly_detection.py \
  code/ISTAD/utils/innovation.py \
  code/ISTAD/scripts/anomaly_detection/ISTAD_v4_hgat_integrated.sh
```

The five hashes must respectively be:

```text
ebf877266b8d61d941396ef27cd0e0fd64e9cc1c9d9fbfe115d6de88e6c39d46
f0482e4c881321e7cb657fd102527b0bca9a62069cc373b36e5907dd8d23ca50
ba0ed839dec4af7650a676e75404c7843a7e77554ac1eb992c84ae4788f10d61
fc13b6fe5a96a0dc7a04ae61a46f3fb9098a7ea84178aeb057a09cc19b47674e
c321f789df214f9a51fc12f5ef8191ea4ab6071150d212691b611249584a4358
```

Back up the only existing file replaced by the patch, then extract:

```bash
cp code/ISTAD/exp/exp_anomaly_detection.py \
   code/ISTAD/exp/exp_anomaly_detection.py.pre_e1_20260911
tar -xzf ISTAD_E1_server_patch_20260911.tar.gz
sha256sum -c E1_PATCHED.sha256
```

## 2. Local logic test

```bash
cd /data/modeluse/TS/ISTAD-release/code/ISTAD
PYTHONPATH=. python -m unittest discover -s tests \
  -p 'test_e1_incidence_audit.py' -v
```

Expected result: five tests pass.

The corrected patch uses the last incidence dimension for the hyperedge count
after the audit tensors have been flattened. Its patched
`exp_anomaly_detection.py` SHA256 is:

```text
5bc771cf8456b0ce832ef5cb4e12c116e45a95e2692b112294b4d82f65aa9cb6
```

## 3. PSM seed-87 smoke run

This reads the frozen checkpoint and uses only three random controls to validate the full path:

```bash
cd /data/modeluse/TS/ISTAD-release/code/ISTAD
ISTAD_E1_RANDOM_REPEATS=3 \
bash scripts/anomaly_detection/ISTAD_v4_hgat_e1.sh PSM 0 87 \
  2>&1 | tee ../../analysis/e1_psm_s87_smoke.log
```

Check the report:

```bash
find ../../analysis/e1_incidence_counterfactual/runtime \
  -name e1_incidence_counterfactual.json -print
python - <<'PY'
import json
from pathlib import Path
p = next(Path('../../analysis/e1_incidence_counterfactual/runtime').rglob(
    'e1_incidence_counterfactual.json'))
r = json.loads(p.read_text())
print('dataset:', r['dataset'])
print('seed:', r['seed'])
print('random_repeats:', r['random_repeats'])
print('frozen_equivalence:', r['frozen_equivalence'])
print('arms:', sorted(r['arms']))
PY
```

The smoke run passes only when `random_repeats` is 3, all four frozen-equivalence differences are at most `1e-12`, and the four expected arms are present.

## 4. Full 12-run audit

At least 3 GB of free temporary disk is recommended. Temporary random-score matrices are deleted after each successful run. Run inside `tmux` or another persistent shell:

```bash
cd /data/modeluse/TS/ISTAD-release
unset ISTAD_E1_RANDOM_REPEATS
bash analysis/run_v4_hgat_e1_all.sh 0 \
  2>&1 | tee analysis/e1_incidence_counterfactual/full_run.log
```

The command runs four datasets and seeds 87, 90, and 98 with 100 random controls, then creates:

```text
analysis/ISTAD_E1_results_20260911.tar.gz
```

Return that archive together with:

```text
analysis/e1_incidence_counterfactual/full_run.log
```

Do not return datasets, checkpoints, temporary mmap files, or the full runtime directory.

## Audit definition

- E1-A: learned dynamic incidence.
- E1-B: entity-wise static normal-train mean incidence.
- E1-C: within-entity time-shuffled learned incidence.
- E1-D: one fixed within-entity node permutation.
- E1-E: 100 independent fixed within-entity node permutations.

E1-E preserves every learned incidence matrix's per-time node-degree profile, sparsity, values, and column normalization exactly. All arms keep feature evidence, the learned train ECDF reference, the reliability decision, rank-safe epsilon, and POT protocol fixed. Test labels enter only after scores and thresholds have been constructed.
