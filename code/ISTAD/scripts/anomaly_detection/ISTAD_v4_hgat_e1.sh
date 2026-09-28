#!/usr/bin/env bash
set -euo pipefail

# Inference-only E1 audit. Existing checkpoints and test_results are never modified.
dataset="${1:-PSM}"
gpu="${2:-0}"
seed="${3:-87}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/../.." && pwd)"
release_root="$(cd "${project_dir}/../.." && pwd)"

export ISTAD_E1_COUNTERFACTUAL=1
export ISTAD_E1_RANDOM_REPEATS="${ISTAD_E1_RANDOM_REPEATS:-100}"
export ISTAD_E1_RESULT_ROOT="${ISTAD_E1_RESULT_ROOT:-${release_root}/analysis/e1_incidence_counterfactual/runtime}"
export ISTAD_RUN_TAG=v4hg

exec bash "${script_dir}/ISTAD_v4_hgat_integrated.sh" \
  "${dataset}" "${gpu}" "${seed}" eval

