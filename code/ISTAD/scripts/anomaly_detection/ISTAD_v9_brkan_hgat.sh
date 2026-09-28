#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/../.." && pwd)"

ISTAD_HGAT_PROJECTION=brkan \
ISTAD_RUN_TAG=v9brk \
ISTAD_CHECKPOINT_ROOT="${project_dir}/../../checkpoints/ISTAD_v9_brkan_hgat" \
exec bash "${script_dir}/ISTAD_v4_hgat_integrated.sh" "$@"
