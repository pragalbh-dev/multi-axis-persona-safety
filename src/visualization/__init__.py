"""Figure rendering. See `src/README.md` for the module map."""

from src.visualization.figures import FIGURE_REGISTRY, figure_paths, make_figure
from src.visualization.types import FigureKind, FigureSpec

__all__ = [
    "FIGURE_REGISTRY",
    "FigureKind",
    "FigureSpec",
    "figure_paths",
    "make_figure",
]
