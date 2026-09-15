#!/usr/bin/env bash
set -euo pipefail
cdlno_inspect_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cdlno_script_root="$(cd -- "$cdlno_inspect_dir/.." && pwd)"

# `path.sh` is user-owned and older copies use aliases such as DATA,
# FNO_DIR and CAR_DATA_DIR.  Source it when present, but never require our
# newer CDLNO_* names to exist.  The inspection script itself must stay tied
# to the checkout containing this file, even when an old path.sh points REPO
# at another Transolver checkout.
if [[ -f "$cdlno_script_root/path.sh" ]]; then
    source "$cdlno_script_root/path.sh"
fi
export CDLNO_REPO_ROOT="${CDLNO_REPO_ROOT:-$cdlno_script_root}"
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
export CDLNO_PYTHON="${CDLNO_PYTHON:-${PYTHON_BIN:-python}}"
exec "$CDLNO_PYTHON" -B "$cdlno_inspect_dir/inspect_data.py" "$@"
