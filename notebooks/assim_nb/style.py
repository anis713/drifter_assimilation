"""Shared visual language for every figure and movie in the notebook.

Colour follows the entity everywhere:
    truth / observations -> ink
    background           -> blue   (categorical slot 1)
    standard analysis    -> orange (slot 2)
    sliding analysis     -> aqua   (slot 3)
Slots 1-3 of the reference palette pass the colour-vision-deficiency checks
for every pair. Speed is a single-hue grey ramp so the coloured
trajectories stay readable on top of it. Drifters are identified by a numbered
label at their release point, not by colour, so any number of drifters works.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:  # plain lon/lat axes still work
    ccrs = cfeature = None
    HAS_CARTOPY = False

# --- surfaces and ink ---------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3de"
LAND = "#e6dcc6"          # warm sand: distinct in hue from the grey speed ramp
COAST = "#9b9993"

# --- entities -----------------------------------------------------------------
TRUTH = INK
BACKGROUND = "#2a78d6"
METHOD_COLOR = {"standard": "#eb6834", "sliding": "#1baf7a"}
METHOD_LABEL = {"standard": "Constant dU per window", "sliding": "Sliding window"}
METHOD_SHORT = {"standard": "constant dU per window", "sliding": "sliding window"}

# --- speed ramp (single hue, light -> dark) -----------------------------------
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "speed_grey", ["#f6f5f2", "#d9d8d2", "#aeada6", "#7d7c76", "#4f4e4a"])

HALO = [pe.withStroke(linewidth=2.6, foreground=SURFACE)]


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "animation.embed_limit": 150.0,   # MB, for inline movies
        "hatch.color": COAST,
        "animation.html": "jshtml",
    })


def method_color(method: str) -> str:
    return METHOD_COLOR[method]


def ordinal_ramp(color: str, n: int) -> list:
    """n steps from a light tint to ``color`` (for ordered windows)."""
    base = np.array(mpl.colors.to_rgb(color))
    light = 0.62 * np.ones(3) + 0.38 * base
    if n == 1:
        return [color]
    return [tuple(light + (base - light) * k / (n - 1)) for k in range(n)]


# --- map axes -------------------------------------------------------------------
def geo(ax) -> dict:
    """Extra kwargs so plot calls work on both cartopy and plain axes."""
    return {"transform": ccrs.PlateCarree()} if HAS_CARTOPY and hasattr(ax, "projection") else {}


def add_map_axes(fig, spec, LON, LAT, left_labels=True, bottom_labels=True):
    """A lon/lat map panel. Land shows through NaN cells of the fields."""
    if HAS_CARTOPY:
        ax = fig.add_subplot(spec, projection=ccrs.PlateCarree())
        ax.set_extent([LON[0], LON[1], LAT[0], LAT[1]], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor=LAND, edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor=COAST, linewidth=0.7, zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color=GRID, alpha=1.0)
        gl.top_labels = gl.right_labels = False
        gl.left_labels = left_labels
        gl.bottom_labels = bottom_labels
        gl.xlabel_style = gl.ylabel_style = {"size": 8, "color": INK_2}
        ax.spines["geo"].set_edgecolor(GRID)
    else:
        ax = fig.add_subplot(spec)
        ax.set_xlim(LON)
        ax.set_ylim(LAT)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(LAT))))
        ax.grid(False)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor(GRID)
        if left_labels:
            ax.set_ylabel("Latitude (°N)")
        else:
            ax.set_yticklabels([])
        if bottom_labels:
            ax.set_xlabel("Longitude (°E)")
        else:
            ax.set_xticklabels([])
    ax.set_facecolor(LAND)   # NaN (land) cells of the speed field read as land
    return ax


def land_mesh(ax, lon, lat, mask):
    """Sand-coloured cells where the field is NaN (land). mask is (nlon, nlat)."""
    Z = np.where(np.asarray(mask).T, 1.0, np.nan)
    return ax.pcolormesh(lon, lat, np.ma.masked_invalid(Z), cmap=mpl.colors.ListedColormap([LAND]),
                         vmin=0, vmax=1, shading="nearest", zorder=2, **geo(ax))


def speed_mesh(ax, lon, lat, U, V, vmax):
    """Speed shading. U, V are (nlon, nlat) as in the tensors."""
    S = np.hypot(U, V).T
    return ax.pcolormesh(lon, lat, np.ma.masked_invalid(S), cmap=SPEED_CMAP,
                         vmin=0.0, vmax=vmax, shading="nearest", zorder=1, **geo(ax))


def quiver_step(lon, lat, target=14):
    return max(1, int(round(max(len(lon), len(lat)) / target)))


def velocity_quiver(ax, lon, lat, U, V, vmax, step):
    X, Y = np.meshgrid(lon[::step], lat[::step])
    u = np.ma.masked_invalid(U[::step, ::step].T)
    v = np.ma.masked_invalid(V[::step, ::step].T)
    return ax.quiver(X, Y, u, v, color=INK, alpha=0.55, scale=vmax * 14, scale_units="width",
                     width=0.0045, headwidth=3.6, headlength=4.2, zorder=4, **geo(ax))


def nice_speed(vmax: float) -> float:
    """A round reference speed for the quiver key."""
    for v in (1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01):
        if v <= 0.6 * vmax:
            return v
    return round(0.6 * vmax, 3)


def label_releases(ax, lon0, lat0):
    """Star + drifter number at every release point."""
    arts = []
    for n, (x, y) in enumerate(zip(lon0, lat0), start=1):
        arts.append(ax.plot(x, y, marker="*", ms=11, color=INK, mec=SURFACE, mew=1.0,
                            zorder=8, ls="none", **geo(ax))[0])
        arts.append(ax.annotate(str(n), (x, y), xytext=(5, 5), textcoords="offset points",
                                fontsize=8.5, fontweight="bold", color=INK, zorder=9,
                                path_effects=HALO))   # PlateCarree data coords = lon/lat
    return arts


def panel_tag(ax, text):
    ax.text(0.015, 0.985, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold", color=INK, zorder=10, path_effects=HALO)


def speed_colorbar(fig, mappable, axes, label="Speed (m s$^{-1}$)"):
    cb = fig.colorbar(mappable, ax=axes, shrink=0.82, pad=0.015, aspect=28)
    cb.set_label(label, color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, length=0)
    return cb


def save_figure(fig, cfg, name: str) -> None:
    if not cfg.save_figures:
        return
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(cfg.fig_dir / f"{name}.{ext}", bbox_inches="tight")
