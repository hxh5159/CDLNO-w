#!/usr/bin/env bash
# Source this file; it only sets paths. No install, mkdir, symlink or training.
# Remote checkout: /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/LinearNO-monitor
# Locating this file keeps the checkout portable (including local --dry-run).
# The CDLNO_* variable names are shared launcher interfaces, not folder names.
# After changing checkouts in an existing shell, explicitly bind this checkout:
#   export CDLNO_REPO_ROOT="$PWD" CDLNO_RUNS_ROOT="$PWD/output"
#   source ./path.sh
# User-supplied path overrides remain supported; data stays at the same root.
export CDLNO_REPO_ROOT="${CDLNO_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export CDLNO_DATA_ROOT="${CDLNO_DATA_ROOT:-/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data}"
export CDLNO_FNO_ROOT="${CDLNO_FNO_ROOT:-${CDLNO_DATA_ROOT}/fno}"

# These are arguments to the ORIGINAL entries, whose path semantics differ.
export CDLNO_DARCY_ROOT="${CDLNO_DARCY_ROOT:-${CDLNO_FNO_ROOT}}"
export CDLNO_ELASTICITY_ROOT="${CDLNO_ELASTICITY_ROOT:-${CDLNO_FNO_ROOT}}"
export CDLNO_NS_ROOT="${CDLNO_NS_ROOT:-${CDLNO_FNO_ROOT}}"
export CDLNO_AIRFOIL_ROOT="${CDLNO_AIRFOIL_ROOT:-${CDLNO_FNO_ROOT}/airfoil/naca}"
export CDLNO_PIPE_ROOT="${CDLNO_PIPE_ROOT:-${CDLNO_FNO_ROOT}/pipe}"
export CDLNO_PLASTICITY_FILE="${CDLNO_PLASTICITY_FILE:-${CDLNO_FNO_ROOT}/plas_N987_T20.mat}"

# Industrial subdirectories are provisional, NOT observed on the remote host.
# Run tran_evaluate/inspect_data.sh and set these to the actual locations.
export CDLNO_CAR_RAW_ROOT="${CDLNO_CAR_RAW_ROOT:-${CDLNO_DATA_ROOT}/mlcfd_data/training_data}"
export CDLNO_CAR_CACHE_ROOT="${CDLNO_CAR_CACHE_ROOT:-${CDLNO_DATA_ROOT}/mlcfd_data/preprocessed_data}"
export CDLNO_AIRFRANS_DATASET="${CDLNO_AIRFRANS_DATASET:-${CDLNO_DATA_ROOT}/AirfRANS/Dataset}"

export CDLNO_RUNS_ROOT="${CDLNO_RUNS_ROOT:-${CDLNO_REPO_ROOT}/output}"
# Change the tag (or explicit run-dir argument) for each new experiment.
# CDLNO_RUN_TAG is optional; each new training invocation gets a timestamp.
export CDLNO_PYTHON="${CDLNO_PYTHON:-python}"
