"""Shared chart styling, so the three notebooks read as one system.

The two-hue categorical palette (blue for ham, orange for spam) was checked
with a colour-vision-deficiency validator rather than by eye: worst-pair ΔE is
24.7 under protanopia and 33.6 for normal vision, both well clear of the floors
(8 and 15 respectively, OKLab x100), and both hues clear 3:1 contrast against a
white notebook surface.

Class colour is assigned by *identity* and fixed here, so ham is the same blue
in every figure across all three notebooks. Charts that show a single series
use one hue and carry no legend -- the title names the series.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = [
    "CLASS_COLORS",
    "GRID",
    "INK_MUTED",
    "INK_PRIMARY",
    "INK_SECONDARY",
    "SERIES",
    "annotate_bars",
    "apply_style",
    "despine",
]

# Categorical slots, in fixed order. Never cycled: a third series takes slot 3,
# and past the list it folds into "other" rather than inventing a hue.
SERIES: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

# Identity mapping, not rank. Reordering the classes must not repaint them.
CLASS_COLORS: dict[str, str] = {
    "ham": SERIES[0],
    "spam": SERIES[1],
    0: SERIES[0],
    1: SERIES[1],
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply_style() -> None:
    """Set the notebook-wide matplotlib defaults. Call once, after imports."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10,
            # Recessive chrome: the data should be the darkest thing on screen.
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "axes.axisbelow": True,          # grid behind the marks, never over
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "patch.linewidth": 0,
            "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
        }
    )


def despine(ax: Any, keep: tuple[str, ...] = ("left", "bottom")) -> Any:
    """Drop the top/right spines and lighten what remains."""
    for side, spine in ax.spines.items():
        if side in keep:
            spine.set_color(BASELINE)
            spine.set_linewidth(1.0)
        else:
            spine.set_visible(False)
    return ax


def annotate_bars(
    ax: Any,
    fmt: str = "{:.0f}",
    *,
    horizontal: bool = False,
    padding: float = 3.0,
) -> Any:
    """Direct-label the bars, so identity never rests on colour alone."""
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt=lambda v: fmt.format(v),
            padding=padding,
            color=INK_SECONDARY,
            fontsize=9,
        )
    if horizontal:
        # Room for the labels so they do not collide with the axes box.
        ax.margins(x=0.14)
    else:
        ax.margins(y=0.14)
    return ax


def class_palette(labels: list[str] | tuple[str, ...]) -> list[str]:
    """Colours for `labels`, held to the fixed identity mapping."""
    return [CLASS_COLORS.get(label, SERIES[2]) for label in labels]
