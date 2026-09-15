#!/usr/bin/env bash
# Darcy CDLNO identity; retain the existing task defaults and checkpoint protocol.
set -euo pipefail
cdlno_ablation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$cdlno_ablation_dir/../../darcy.sh" eval --front-latent-mode identity "$@"
