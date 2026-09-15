#!/usr/bin/env bash
# Darcy CDLNO no_sa; retain the existing task defaults and checkpoint protocol.
set -euo pipefail
cdlno_ablation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$cdlno_ablation_dir/../../darcy.sh" train --front-latent-mode no_sa "$@"
