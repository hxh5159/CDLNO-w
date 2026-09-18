"""Publication-friendly heatmaps for LinearNO layer-kernel similarity."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_snapshot_plots(snapshot: Path, mean: np.ndarray, std: np.ndarray) -> None:
    """Write PNG/PDF/SVG heatmaps when matplotlib is installed."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        (snapshot / "plot_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if mean.ndim != 2 or mean.shape[0] != mean.shape[1] or std.shape != mean.shape:
        raise ValueError(f"expected square mean/std matrices, got {mean.shape}/{std.shape}")
    layers = mean.shape[0]
    with plt.rc_context({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), squeeze=False)
        axes = axes[0]
        for axis, matrix, title in ((axes[0], mean, r"$\rho_{ij}$ mean"),
                                    (axes[1], std, r"$\rho_{ij}$ std")):
            image = axis.imshow(matrix, origin="lower", cmap="viridis",
                                vmin=0.0 if axis is axes[0] else None,
                                vmax=1.0 if axis is axes[0] else None,
                                interpolation="nearest", aspect="equal")
            axis.set_title(title)
            axis.set_xlabel("Layer j")
            axis.set_ylabel("Layer i")
            axis.set_xticks(range(layers))
            axis.set_yticks(range(layers))
            axis.set_xticklabels([str(i + 1) for i in range(layers)])
            axis.set_yticklabels([str(i + 1) for i in range(layers)])
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.suptitle("LinearNO propagation-kernel similarity")
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(snapshot / f"similarity_heatmap.{suffix}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)

