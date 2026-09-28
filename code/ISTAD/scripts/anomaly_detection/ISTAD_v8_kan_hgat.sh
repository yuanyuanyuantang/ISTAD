#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/../.." && pwd)"

ISTAD_HGAT_PROJECTION=kan \
ISTAD_RUN_TAG=v8khg \
ISTAD_CHECKPOINT_ROOT="${project_dir}/../../checkpoints/ISTAD_v8_kan_hgat" \
exec bash "${script_dir}/ISTAD_v4_hgat_integrated.sh" "$@"
