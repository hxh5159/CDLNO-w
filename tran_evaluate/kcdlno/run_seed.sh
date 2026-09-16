#!/usr/bin/env bash
# One selected seed; each task trains, evaluates and reports before the next.
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: bash tran_evaluate/kcdlno/run_seed.sh 0|1|2 [--gpu 0] [--dry-run]
       bash tran_evaluate/kcdlno/run_seed.sh --seed 0|1|2 [--gpu 0] [--dry-run]
Order: darcy -> airfoil -> plas (plasticity) -> elas (elasticity) -> ns -> pipe.
Each task: train -> eval -> print results with seed -> next task.
No arguments: ask for one seed in an interactive terminal.
Uses the existing KCDNO task presets and path.sh; no automatic seed sweep.
EOF
}

for arg in "$@"; do
    case "$arg" in -h|--help) usage; exit 0 ;; esac
done
if (($# == 0)); then
    if [[ -t 0 ]]; then
        read -r -p 'Choose seed (0, 1, or 2): ' selected_seed
        set -- --seed "$selected_seed"
    else
        usage >&2
        exit 2
    fi
fi
case "$1" in 0|1|2) selected_seed="$1"; shift; set -- --seed "$selected_seed" "$@" ;; esac
source "$launcher/../_common.sh"
exec "$CDLNO_PYTHON" -u "$launcher/_seed_suite.py" "$@"
