#!/usr/bin/env bash
set -euo pipefail

task="${1:?dataset task is required}"
shift

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
launcher="$repo_root/tran_evaluate/linearno_loop_v4/${task}.sh"
[[ -x "$launcher" ]] || { echo "v4 launcher not found: $launcher" >&2; exit 2; }

# Load the checkout's path contract before deriving task-specific arguments.
# Existing environment overrides remain authoritative because path.sh uses
# ${VAR:-default} assignments.
if [[ -f "$repo_root/path.sh" ]]; then
  # shellcheck disable=SC1091
  source "$repo_root/path.sh"
fi

# With no action, the dataset wrapper means train -> evaluate.  The existing
# v4 launcher still handles resume/eval/dry-run and all saved-metadata checks.
action="train_eval"
if [[ $# -gt 0 && "$1" != -* ]]; then
  action="$1"
  shift
fi

case "$action" in
  train|train_eval)
    has_mode=0
    for arg in "$@"; do
      case "$arg" in
        --temperature-mode|--temperature-mode=*) has_mode=1 ;;
      esac
    done
    if [[ "$has_mode" -eq 0 ]]; then
      # Recommended full-budget experiment: pointwise Q and latent-wise K.
      # Override with V4_TEMPERATURE_MODE or pass --temperature-mode explicitly.
      set -- --temperature-mode "${V4_TEMPERATURE_MODE:-latent_k_point_q}" "$@"
    fi
    ;;
  resume|eval) ;;
  *)
    echo "usage: $0 [train|train_eval|resume|eval] [v4 launcher options]" >&2
    exit 2
    ;;
esac

# The original PDE entries have task-specific data_path contracts.  Supplying
# the task path here also prevents the generic v4 data-root fallback from
# becoming the last (and therefore effective) duplicate --data_path option.
has_data_path=0
for arg in "$@"; do
  case "$arg" in
    --data_path|--data_path=*|--data-path|--data-path=*) has_data_path=1 ;;
  esac
done
if [[ "$has_data_path" -eq 0 ]]; then
  data_root="${CDLNO_DATA_ROOT:-$repo_root/../data}"
  fno_root="${CDLNO_FNO_ROOT:-$data_root/fno}"
  case "$task" in
    airfoil) data_path="${CDLNO_AIRFOIL_ROOT:-$fno_root/airfoil/naca}" ;;
    darcy) data_path="${CDLNO_DARCY_ROOT:-$fno_root}" ;;
    elasticity) data_path="${CDLNO_ELASTICITY_ROOT:-$fno_root}" ;;
    pipe) data_path="${CDLNO_PIPE_ROOT:-$fno_root/pipe}" ;;
    *) data_path="" ;;
  esac
  [[ -z "$data_path" ]] || set -- --data_path "$data_path" "$@"
fi

exec "$launcher" "$action" "$@"
