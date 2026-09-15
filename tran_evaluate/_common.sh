#!/usr/bin/env bash
# Shared command dispatch only; no model, loader, optimizer or time-loop code.

cdlno_common_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$cdlno_common_root/path.sh" ]]; then
    source "$cdlno_common_root/path.sh"
fi
# Keep execution anchored to the checkout containing this dispatcher.  A
# legacy path.sh may expose REPO=...Transolver_re (or an invalid CDLNO root),
# which must not redirect the new scripts to a different tree.
cdlno_repo_root="$cdlno_common_root"
if [[ -n "${CDLNO_REPO_ROOT:-}" && -d "${CDLNO_REPO_ROOT}/tran_evaluate" ]]; then
    cdlno_repo_root="$CDLNO_REPO_ROOT"
fi

# Compatibility aliases used by the remote path.sh currently in circulation.
export CDLNO_DATA_ROOT="${CDLNO_DATA_ROOT:-${DATA:-}}"
export CDLNO_FNO_ROOT="${CDLNO_FNO_ROOT:-${FNO_DIR:-${CDLNO_DATA_ROOT:+$CDLNO_DATA_ROOT/fno}}}"
export CDLNO_DARCY_ROOT="${CDLNO_DARCY_ROOT:-${CDLNO_FNO_ROOT:-}}"
export CDLNO_ELASTICITY_ROOT="${CDLNO_ELASTICITY_ROOT:-${CDLNO_FNO_ROOT:-}}"
export CDLNO_NS_ROOT="${CDLNO_NS_ROOT:-${CDLNO_FNO_ROOT:-}}"
export CDLNO_AIRFOIL_ROOT="${CDLNO_AIRFOIL_ROOT:-${CDLNO_FNO_ROOT:+$CDLNO_FNO_ROOT/airfoil/naca}}"
export CDLNO_PIPE_ROOT="${CDLNO_PIPE_ROOT:-${CDLNO_FNO_ROOT:+$CDLNO_FNO_ROOT/pipe}}"
export CDLNO_PLASTICITY_FILE="${CDLNO_PLASTICITY_FILE:-${CDLNO_FNO_ROOT:+$CDLNO_FNO_ROOT/plas_N987_T20.mat}}"
export CDLNO_CAR_RAW_ROOT="${CDLNO_CAR_RAW_ROOT:-${CAR_DATA_DIR:-}}"
export CDLNO_CAR_CACHE_ROOT="${CDLNO_CAR_CACHE_ROOT:-${CAR_SAVE_DIR:-}}"
export CDLNO_AIRFRANS_DATASET="${CDLNO_AIRFRANS_DATASET:-${AIRFRANS_DATASET:-}}"
export CDLNO_RUNS_ROOT="${CDLNO_RUNS_ROOT:-$cdlno_repo_root/output}"
# CDLNO_RUN_TAG is optional; each new training invocation gets a timestamp.
export CDLNO_PYTHON="${CDLNO_PYTHON:-${PYTHON_BIN:-python}}"

cdlno_front_run_suffix() {
    # Only choose the implicit run name; explicit run paths still win last.
    # Do not inject a mode flag: eval with an explicit run can infer its mode.
    local selected=full
    while (($#)); do
        case "$1" in
            --front-latent-mode|--front_latent_mode)
                (($# >= 2)) || { printf 'Missing front latent mode\n' >&2; return 2; }
                selected="$2"; shift ;;
            --front-latent-mode=*|--front_latent_mode=*) selected="${1#*=}" ;;
        esac
        shift
    done
    case "$selected" in
        full) ;;
        no_sa|identity) printf '_%s' "$selected" ;;
        *) printf 'Invalid front_latent_mode: %s\n' "$selected" >&2; return 2 ;;
    esac
}

cdlno_run_path() {
    local task="$1" action="$2" selected="" flag
    shift 2
    while (($#)); do
        flag="$1"; shift
        case "$flag" in
            --cdlno-run-dir|--run_dir)
                (($#)) || { printf 'Missing run directory\n' >&2; return 2; }
                selected="$1"; shift ;;
            --cdlno-run-dir=*|--run_dir=*) selected="${flag#*=}" ;;
        esac
    done
    if [[ -n "$selected" ]]; then
        printf '%s' "$selected"
    elif [[ -n "${CDLNO_RUN_TAG:-}" ]]; then
        printf '%s/%s/%s' "$CDLNO_RUNS_ROOT" "$task" "$CDLNO_RUN_TAG"
    elif [[ "$action" == train ]]; then
        printf '%s/%s/%s' "$CDLNO_RUNS_ROOT" "$task" "$(date -u +%Y%m%dT%H%M%S%6NZ)"
    else
        printf 'Evaluation needs --cdlno-run-dir/--run_dir or an explicit CDLNO_RUN_TAG; no latest-run guessing.\n' >&2
        return 2
    fi
}

cdlno_dispatch() {
    local task_project="$1" task_entry="$2"
    shift 2
    local task_arg task_flag task_value task_dry_run=0
    local -a task_args=()
    # Resolve user paths before changing to the original project working directory.
    while (($#)); do
        task_arg="$1"
        shift
        case "$task_arg" in
            --dry-run) task_dry_run=1 ;;
            --data_dir|--save_dir|--run_dir|--data_path|--cdlno-run-dir|--my_path|--save_path)
                if (($# == 0)) || [[ -z "$1" || "$1" == --* ]]; then
                    printf 'Missing value for %s\n' "$task_arg" >&2
                    return 2
                fi
                task_value="$1"
                shift
                [[ "$task_value" = /* ]] || task_value="$PWD/$task_value"
                task_args+=("$task_arg" "$task_value")
                ;;
            --data_dir=*|--save_dir=*|--run_dir=*|--data_path=*|--cdlno-run-dir=*|--my_path=*|--save_path=*)
                task_flag="${task_arg%%=*}"
                task_value="${task_arg#*=}"
                if [[ -z "$task_value" ]]; then
                    printf 'Missing value for %s\n' "$task_flag" >&2
                    return 2
                fi
                [[ "$task_value" = /* ]] || task_value="$PWD/$task_value"
                task_args+=("$task_flag=$task_value")
                ;;
            *) task_args+=("$task_arg") ;;
        esac
    done
    local task_python="${CDLNO_PYTHON:-python}"
    export PYTHONPATH="${cdlno_repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
    # -u preserves prompt progress/log output; plotting still uses original code.
    export MPLBACKEND="${MPLBACKEND:-Agg}"
    printf 'Working directory: %s\n' "$cdlno_repo_root/$task_project"
    printf 'Command:'
    printf ' %q' "$task_python" -u "$task_entry" "${task_args[@]}"
    printf '\n'
    if ((task_dry_run)); then
        printf 'DRY RUN: command only; no Python entry, data, training or evaluation executed.\n'
        return 0
    fi
    if [[ "$task_project" == Car-Design-ShapeNetCar && "$task_entry" == main_evaluation.py ]]; then
        "$task_python" -B "$cdlno_repo_root/tran_evaluate/check_car_eval.py" "${task_args[@]}" || return $?
    fi
    cd "$cdlno_repo_root/$task_project"
    exec "$task_python" -u "$task_entry" "${task_args[@]}"
}
