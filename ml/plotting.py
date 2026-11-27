"""Colours shared by the figures.

The gain classes use a palette that stays distinguishable with colour-vision
deficiencies: negative = vermilion, neutral = grey, positive = blue.
"""
import numpy as np
from matplotlib.colors import to_hex, to_rgb

NEGATIVE_COLOR = "#D55E00"   # vermilion (Okabe-Ito)
NEUTRAL_COLOR = "#B0B0B0"    # grey
POSITIVE_COLOR = "#2B82C5"   # blue
CLASS_COLORS = [NEGATIVE_COLOR, NEUTRAL_COLOR, POSITIVE_COLOR]


def on_white(color, alpha):
    """Colour seen when ``color`` is drawn with ``alpha`` over white."""
    return to_hex(alpha * np.array(to_rgb(color)) + (1 - alpha) * np.ones(3))
