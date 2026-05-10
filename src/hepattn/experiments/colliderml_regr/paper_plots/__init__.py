"""Unified paper-plot pipeline.

One CLI invocation per training run produces a self-contained reproducibility
bundle (config copy, checkpoint symlink, predictions h5, plots in PDF + PNG,
stats). All output paths are env-var-overrideable; defaults resolve to
``logs/paper_plots/`` and ``logs/comet_offline/`` relative to the package
root, with the dataset path read from ``DATA_ROOT``.
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt


# Repo root: this file is at src/hepattn/experiments/colliderml_regr/paper_plots/__init__.py
# so 5 parents up is the package root.
_REPO_ROOT = Path(__file__).resolve().parents[5]

PAPER_PLOTS_ROOT = Path(os.environ.get("PAPER_PLOTS_ROOT", _REPO_ROOT / "logs" / "paper_plots"))
DATA_DIR = Path(os.environ.get("DATA_ROOT", "/data/colliderml")) / "p200_core_kf_matched_finetune"
COMET_OFFLINE_ROOT = Path(os.environ.get("COMET_OFFLINE_ROOT", _REPO_ROOT / "logs" / "comet_offline"))


def apply_paper_style() -> None:
    """Set rcParams for paper-grade figures."""
    mpl.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # editable text in vector PDF
        "ps.fonttype": 42,
    })


def save_fig(fig, output_dir: Path | str, stem: str) -> None:
    """Save figure as both PDF (vector) and PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)
