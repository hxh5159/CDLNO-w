#!/usr/bin/env bash
# New-family path flags, using the same installed Python and original cwd.
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$launcher/../_common.sh"
project="$1"; entry="$2"; shift 2
dry=0; args=()
while (($#)); do
    arg="$1"; shift
    case "$arg" in
        --dry-run) dry=1 ;;
        --msar-run-dir|--run_dir|--data_dir|--save_dir|--my_path|--save_path)
            value="${1:?missing path}"; shift
            [[ "$value" = /* ]] || value="$PWD/$value"
            args+=("$arg" "$value") ;;
        --msar-run-dir=*|--run_dir=*|--data_dir=*|--save_dir=*|--my_path=*|--save_path=*)
            value="${arg#*=}"; [[ "$value" = /* ]] || value="$PWD/$value"
            args+=("${arg%%=*}" "$value") ;;
        *) args+=("$arg") ;;
    esac
done
export PYTHONPATH="$cdlno_repo_root${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
printf 'Working directory: %s\n' "$cdlno_repo_root/$project"
printf 'Command:'; printf ' %q' "$CDLNO_PYTHON" -u "$entry" "${args[@]}"; printf '\n'
((dry)) && { printf 'DRY RUN: no Python/data/training/evaluation.\n'; exit 0; }
cd "$cdlno_repo_root/$project"
if [[ "$project" == Car-Design-ShapeNetCar && "$entry" == main_evaluation.py ]]; then
    "$CDLNO_PYTHON" -B -m cdlno.msar_lno.car_preflight "${args[@]}"
fi
exec "$CDLNO_PYTHON" -u "$entry" "${args[@]}"
