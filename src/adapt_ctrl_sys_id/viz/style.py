"""Shared visual identity for figures, diagrams and animations."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

__all__ = ["COLORS", "apply_style", "header", "save"]

COLORS = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "faint": "#94a3b8",
    "grid": "#e2e8f0",
    "panel": "#f8fafc",
    "measured": "#2563eb",
    "model": "#f97316",
    "lab": "#94a3b8",
    "alt": "#a855f7",
    "good": "#10b981",
    "bad": "#ef4444",
    "reference": "#1e293b",
    "effort": "#8b5cf6",
    "accent": "#0ea5e9",
    "gold": "#f59e0b",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["faint"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 8.0,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
            "svg.fonttype": "path",
        }
    )


def header(fig: Figure, title: str, subtitle: str | None = None) -> None:
    """Left-aligned title and grey subtitle placed just above the finished layout."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes if ax.get_visible()]
    top = max(b.y1 for b in boxes)
    left = min(b.x0 for b in boxes)
    width, height = fig.bbox.width, fig.bbox.height
    pt = fig.dpi / 72.0
    y = top + 9.0 * pt
    if subtitle:
        fig.text(left / width, y / height, subtitle, fontsize=10.5, color=COLORS["muted"], ha="left", va="bottom")
        y += 19.0 * pt
    fig.text(
        left / width, y / height, title, fontsize=15, fontweight="bold", color=COLORS["ink"], ha="left", va="bottom"
    )


def save(fig: Figure, path: Path, title: str | None = None, subtitle: str | None = None, dpi: int = 150) -> Path:
    """Add the header (if any) and write the figure."""
    if title:
        header(fig, title, subtitle)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
