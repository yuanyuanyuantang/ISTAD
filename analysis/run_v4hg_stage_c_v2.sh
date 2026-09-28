#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
DATA_ROOT="${2:-/data/modeluse/TS/ISTAD/dataset}"
DEVICE="${3:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$MODE" in
  smoke)
    DATASETS=(EXA)
    OUT_REL="analysis/v4hg_interpretability/stage_c_v2_smoke"
    ARCHIVE_REL="analysis/ISTAD_StageC_v2_smoke_results_20260915.tar.gz"
    ;;
  full)
    DATASETS=(EXA PSM SMD SWAT)
    OUT_REL="analysis/v4hg_interpretability/stage_c_v2"
    ARCHIVE_REL="analysis/ISTAD_StageC_v2_full_results_20260915.tar.gz"
    ;;
  *)
    echo "Usage: $0 {smoke|full} [data-root] [device]" >&2
    exit 2
    ;;
esac

cd "$ROOT"
OUT="$ROOT/$OUT_REL"
ARCHIVE="$ROOT/$ARCHIVE_REL"
if [[ -e "$OUT/stability_v2.json" || -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite an existing Stage C v2 result or archive." >&2
  exit 3
fi
mkdir -p "$OUT"

EXPECTED_OLD="143fcc29b816c63ce4f87e338cb8058a7603fc40077ff704381cc2ce6361a68c"
ACTUAL_OLD="$(sha256sum analysis/v4hg_interpretability/stability.json | awk '{print $1}')"
if [[ "$ACTUAL_OLD" != "$EXPECTED_OLD" ]]; then
  echo "Original stability.json hash mismatch before the run." >&2
  exit 4
fi

echo "Stage C v2 mode=$MODE datasets=${DATASETS[*]} device=$DEVICE"
"$PYTHON_BIN" analysis/test_stage_c_v2_metrics.py 2>&1 \
  | tee "$OUT/unit_tests.log"

"$PYTHON_BIN" -u analysis/evaluate_v4hg_stability_v2.py \
  --data-root "$DATA_ROOT" \
  --device "$DEVICE" \
  --datasets "${DATASETS[@]}" \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

"$PYTHON_BIN" analysis/verify_v4hg_stability_v2.py \
  --repo-root "$ROOT" \
  --result-dir "$OUT" \
  --datasets "${DATASETS[@]}" \
  2>&1 | tee "$OUT/verification.log"

ACTUAL_OLD_AFTER="$(sha256sum analysis/v4hg_interpretability/stability.json | awk '{print $1}')"
if [[ "$ACTUAL_OLD_AFTER" != "$EXPECTED_OLD" ]]; then
  echo "Original stability.json changed during the run." >&2
  exit 5
fi
if find "$OUT" -type d -name '_scratch' -print -quit | grep -q .; then
  echo "Temporary incidence directory was not cleaned." >&2
  exit 6
fi

(
  cd "$OUT"
  sha256sum stability_v2.json STABILITY_V2_RESULTS.md VERIFICATION.json \
    unit_tests.log run.log verification.log > SHA256SUMS.txt
)

tar -czf "$ARCHIVE" \
  "$OUT_REL" \
  analysis/evaluate_v4hg_stability_v2.py \
  analysis/stage_c_v2_metrics.py \
  analysis/test_stage_c_v2_metrics.py \
  analysis/verify_v4hg_stability_v2.py \
  analysis/run_v4hg_stage_c_v2.sh \
  analysis/v4hg_interpretability/STAGE_C_V2_PROTOCOL_AMENDMENT.md

sha256sum "$ARCHIVE" | tee "${ARCHIVE}.sha256"
echo "Stage C v2 $MODE completed and verified: $ARCHIVE"
