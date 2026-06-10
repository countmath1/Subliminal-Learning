"""Shared matplotlib style for paper-quality figures.

Call `setup_style()` at the top of any plot script. Gives:
  - Serif body text (Computer Modern when available, falls back gracefully)
  - Math rendered via matplotlib's mathtext in CM style (no system LaTeX)
  - Paper-appropriate font sizes (~9-10 pt)
  - Removed top/right spines, light y-axis grid
  - Higher DPI output

If you ever do install a full LaTeX distribution and want truly identical
output to a TeX document, flip `text.usetex` to True — but for screenshots
and laptop-side iteration, mathtext is usually indistinguishable.
"""
import matplotlib as mpl


def setup_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "Latin Modern Roman",
            "Liberation Serif",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "mathtext.default": "regular",

        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,

        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,

        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,

        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "grid.color": "gray",

        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
