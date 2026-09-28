#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/../.." && pwd)"

ISTAD_HGAT_PROJECTION=brkan \
ISTAD_HGAT_INPUT=innovation \
ISTAD_HGAT_FUSION_STRATEGY=reliability_mix \
ISTAD_RUN_TAG=v10ibrk \
ISTAD_CHECKPOINT_ROOT="${project_dir}/../../checkpoints/ISTAD_v10_innovation_brkan_hgat" \
exec bash "${script_dir}/ISTAD_v4_hgat_integrated.sh" "$@"
