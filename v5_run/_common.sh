#!/usr/bin/env bash
set -euo pipefail

task="${1:?dataset task is required}"
default_gpu="${2:?default GPU is required}"
shift 2

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
launcher="$repo_root/tran_evaluate/linearno_loop_v5/${task}.sh"

[[ -f "$launcher" ]] || {
    printf 'V5 launcher not found: %s\n' "$launcher" >&2
    exit 2
}

export CDLNO_REPO_ROOT="$repo_root"
if [[ -f "$repo_root/path.sh" ]]; then
    # shellcheck disable=SC1091
    source "$repo_root/path.sh"
fi
export CDLNO_REPO_ROOT="$repo_root"
export CDLNO_RUNS_ROOT="${V5_RUNS_ROOT:-${CDLNO_RUNS_ROOT:-$repo_root/output}}"

# With no action, run training followed by evaluation of the same run.
action="train_eval"
if (($#)) && [[ "$1" != --* ]]; then
    action="$1"
    shift
fi

case "$action" in
    train|train_eval|resume|eval) ;;
    dry-run|preview)
        action="train"
        set -- --dry-run "$@"
        ;;
    print-run-dir)
        action="train"
        set -- --print-run-dir "$@"
        ;;
    *)
        printf 'Usage: %s [train|train_eval|resume|eval|dry-run|preview|print-run-dir] [options]\n' "$0" >&2
        exit 2
        ;;
esac

has_option() {
    local option="$1"
    shift
    local argument
    for argument in "$@"; do
        case "$argument" in
            "$option"|"$option="*) return 0 ;;
        esac
    done
    return 1
}

defaults=()

# Resume/eval recover architecture fields from immutable saved metadata. Only
# new training receives experiment defaults; user options remain last and win.
if [[ "$action" == "train" || "$action" == "train_eval" ]]; then
    if ! has_option --topology "$@" && ! has_option --executed-depth "$@"; then
        defaults+=(--topology "${V5_TOPOLOGY:-p1_c3_r2_s1}")
    fi
    if ! has_option --expert-count "$@"; then
        defaults+=(--expert-count "${V5_EXPERT_COUNT:-3}")
    fi
    if ! has_option --core-norm-mode "$@"; then
        defaults+=(--core-norm-mode "${V5_CORE_NORM_MODE:-visit_independent}")
    fi
    if [[ -n "${V5_EXPERT_WIDTH:-}" ]] && ! has_option --expert-width "$@"; then
        defaults+=(--expert-width "$V5_EXPERT_WIDTH")
    fi
    if ! has_option --linearno-profile "$@"; then
        defaults+=(--linearno-profile "${V5_LINEARNO_PROFILE:-paper_table8_on_release_model}")
    fi
    if ! has_option --seed "$@"; then
        defaults+=(--seed "${SEED:-0}")
    fi
fi

if ! has_option --gpu "$@"; then
    defaults+=(--gpu "${GPU:-$default_gpu}")
fi

exec bash "$launcher" "$action" "${defaults[@]}" "$@"
