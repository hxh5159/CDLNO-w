#!/usr/bin/env bash
# Shared command dispatch only; no model, loader, optimizer or time-loop code.

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
            --data_dir|--save_dir|--run_dir|--data_path|--cdlno-run-dir)
                if (($# == 0)) || [[ -z "$1" || "$1" == --* ]]; then
                    printf 'Missing value for %s\n' "$task_arg" >&2
                    return 2
                fi
                task_value="$1"
                shift
                [[ "$task_value" = /* ]] || task_value="$PWD/$task_value"
                task_args+=("$task_arg" "$task_value")
                ;;
            --data_dir=*|--save_dir=*|--run_dir=*|--data_path=*|--cdlno-run-dir=*)
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
