#!/usr/bin/env bash
set -euo pipefail

# Kept as a compatibility entry point.  New runs should use ISTAD_v3_strict.sh.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/ISTAD_v3_strict.sh" "$@"
