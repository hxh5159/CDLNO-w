#!/usr/bin/env python3
"""Regenerate LinearNO heatmaps from ``similarity_values.npz``."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .plotting import write_snapshot_plots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path,
                        help="snapshot directory containing similarity_values.npz")
    args = parser.parse_args()
    snapshot = args.snapshot.expanduser().resolve()
    with np.load(snapshot / "similarity_values.npz", allow_pickle=False) as values:
        write_snapshot_plots(snapshot, values["rho_mean"], values["rho_std"])
    print(f"plots written to {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
