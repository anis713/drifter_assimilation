"""Static figures: domain overview, error versus time, objective decay, summary."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from . import results as R
from . import style as S


def nanmean0(a):
    """Time mean that stays quiet on all-NaN (land) cells."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=0)


# --------------------------------------------------------------------------- #
# Figure 1: domain, mean currents and release points
# --------------------------------------------------------------------------- #
def plot_domain_overview(cfg, lon, lat, Ub, Vb, Ut, Vt):
    """Time-mean background and truth over the assimilation period.

    Ub, Vb, Ut, Vt: arrays (time, nlon, nlat); lon (nlon,), lat (nlat,).
    """
    lon, lat = np.asarray(lon), np.asarray(lat)
    mean = {k: nanmean0(a) for k, a in dict(Ub=Ub, Vb=Vb, Ut=Ut, Vt=Vt).items()}
    vmax = float(np.nanpercentile(np.hypot(np.r_[mean["Ub"], mean["Ut"]],
                                           np.r_[mean["Vb"], mean["Vt"]]), 99))
    step = S.quiver_step(lon, lat)

    fig = plt.figure(figsize=(11.5, 5.0), layout="constrained")
    gs = GridSpec(1, 2, figure=fig)
    axes, mesh = [], None
    panels = [("a  Background (altimetry)", mean["Ub"], mean["Vb"]),
              ("b  Truth (reanalysis)", mean["Ut"], mean["Vt"])]
    for i, (title, U, V) in enumerate(panels):
        ax = S.add_map_axes(fig, gs[0, i], cfg.lon, cfg.lat, left_labels=(i == 0))
        mesh = S.speed_mesh(ax, lon, lat, U, V, vmax)
        S.land_mesh(ax, lon, lat, np.isnan(U))
        q = S.velocity_quiver(ax, lon, lat, U, V, vmax, step)
        S.label_releases(ax, cfg.lon0, cfg.lat0)
        S.panel_tag(ax, title)
        axes.append(ax)
    ref = S.nice_speed(vmax)
    axes[-1].quiverkey(q, 0.80, 1.03, ref, f"{ref:g} m s$^{{-1}}$", labelpos="E",
                       coordinates="axes", color=S.INK, labelcolor=S.INK_2,
                       fontproperties={"size": 8})
    S.speed_colorbar(fig, mesh, axes)
    fig.suptitle(f"Mean surface currents, days {cfg.start_day}–{cfg.start_day + cfg.n_days}"
                 f"  ·  {cfg.n_drif} numbered release point{'s' if cfg.n_drif > 1 else ''}",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=S.INK)
    return fig


# --------------------------------------------------------------------------- #
# Figure 2: field error over the assimilation window
# --------------------------------------------------------------------------- #
def _window_strip(ax, cfg, spans_by_method):
    """Gantt-style rows showing each method's correction windows."""
    rows = list(spans_by_method)
    y = 0.0
    ticks = []
    for method in rows:
        spans = spans_by_method[method]
        color = S.method_color(method)
        layers = 1
        if method == "sliding" and cfg.tw_days > cfg.sigma_days:
            layers = math.ceil(cfg.tw_days / cfg.sigma_days)
        h = 0.8 / layers
        for k, (a, b) in enumerate(spans):
            yy = y + (layers - 1 - k % layers) * h
            ax.broken_barh([(a + 0.015, (b - a) - 0.03)], (yy + 0.06 * h, 0.88 * h),
                           facecolor=color, edgecolor="none", alpha=0.9)
            if b - a >= 0.6:
                ax.text(0.5 * (a + b), yy + 0.5 * h, f"{k + 1}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        ticks.append(y + 0.4)
        y += 1.0
    ax.set_yticks(ticks, [S.METHOD_LABEL[m] for m in rows], fontsize=8.5)
    ax.set_ylim(-0.1, y - 0.1)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(labelbottom=False)
    ax.set_title("Correction windows", fontsize=9.5, color=S.INK_2, fontweight="normal")


def plot_rmse_time(cfg, rms_by_method, spans_by_method):
    """Relative field RMSE of background and analyses, and its reduction ΔRMSE(t).

    rms_by_method: {method: results.load_rms(...)}
    spans_by_method: {method: results.window_spans(...)}
    """
    methods = list(rms_by_method)
    fig = plt.figure(figsize=(10, 6.6), layout="constrained")
    gs = GridSpec(3, 1, figure=fig, height_ratios=[0.22 * len(methods) + 0.12, 1.0, 0.85])
    ax_w = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax_w)
    ax_d = fig.add_subplot(gs[2], sharex=ax_w)

    _window_strip(ax_w, cfg, spans_by_method)

    first = rms_by_method[methods[0]]
    t = first["t_days"]
    ax_r.plot(t, first["back"], color=S.BACKGROUND, lw=2.0, label="Background")
    for m in methods:
        r = rms_by_method[m]
        ax_r.plot(r["t_days"], r["corr"], color=S.method_color(m), lw=2.0,
                  label=f"Analysis · {S.METHOD_SHORT[m]}")
        ax_d.plot(r["t_days"], r["drms"], color=S.method_color(m), lw=2.0)
        ax_d.annotate(f"mean {r['rel']:.1f} %", (r["t_days"][-1], r["drms"][-1]),
                      xytext=(6, 0), textcoords="offset points", va="center",
                      fontsize=8.5, color=S.INK_2, annotation_clip=False)
    ax_r.set_ylabel("Relative RMSE (%)")
    ax_r.set_ylim(bottom=0)
    ax_r.legend(loc="upper left", ncols=len(methods) + 1, bbox_to_anchor=(0, 1.13))
    ax_r.tick_params(labelbottom=False)

    ax_d.axhline(0, color=S.MUTED, lw=0.8)
    ax_d.set_ylabel("RMSE reduction (%)")
    ax_d.set_xlabel("Time since start of assimilation (days)")
    ax_d.set_xlim(t[0], cfg.n_days)
    for ax in (ax_r, ax_d):
        for d in range(1, cfg.n_days):
            ax.axvline(d, color=S.GRID, lw=0.8, zorder=0)

    fig.suptitle("Velocity-field error over the assimilation window",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=S.INK)
    return fig


# --------------------------------------------------------------------------- #
# Figure 3: objective function decay
# --------------------------------------------------------------------------- #
def plot_objective(cfg, obj_by_method):
    """Cost function per Adam iteration, one curve per window, log scale."""
    methods = list(obj_by_method)
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 3.9),
                             layout="constrained", squeeze=False)
    for ax, m in zip(axes[0], methods):
        curves = obj_by_method[m]
        colors = S.ordinal_ramp(S.method_color(m), len(curves))
        for k, (J, c) in enumerate(zip(curves, colors)):
            it = np.arange(1, len(J) + 1)
            ax.plot(it, J, color=c, lw=1.8, label=f"window {k + 1}")
            ax.plot(it[-1], J[-1], "o", ms=4.5, color=c, mec=S.SURFACE, mew=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Adam iteration")
        ax.set_title(S.METHOD_LABEL[m])
        if len(curves) <= 8:
            ax.legend(loc="upper right")
    axes[0][0].set_ylabel("Objective function J")
    fig.suptitle(r"Objective-function decay  (trajectory misfit $+\,\alpha\,\|\delta u\| + \beta\,$divergence)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=S.INK)
    return fig


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
def summary_table(cfg, h5):
    rows = []
    for m in cfg.methods:
        if m not in h5:
            continue
        r = R.load_rms(h5, m, cfg)
        its = R.iterations(h5, m, cfg)
        rows.append({
            "Method": S.METHOD_LABEL[m],
            "Windows": len(its),
            "Iterations per window": ", ".join(map(str, its)),
            "Mean RMSE, background (%)": r["back"].mean(),
            "Mean RMSE, analysis (%)": r["corr"].mean(),
            "Relative ΔRMSE (%)": r["rel"],
            "Runtime (s)": h5[m].attrs.get("runtime_s", np.nan),
        })
    df = pd.DataFrame(rows).set_index("Method")
    return (df.style
              .format({"Mean RMSE, background (%)": "{:.2f}", "Mean RMSE, analysis (%)": "{:.2f}",
                       "Relative ΔRMSE (%)": "{:.2f}", "Runtime (s)": "{:.0f}"})
              .set_properties(**{"font-family": "Helvetica Neue, Helvetica, Arial, sans-serif",
                                 "font-size": "10pt"}))


def trajectory_legend_handles(method):
    return [
        Line2D([], [], color=S.TRUTH, lw=2.0, label="Observed (truth)"),
        Line2D([], [], color=S.BACKGROUND, lw=1.8, label="Background"),
        Line2D([], [], color=S.method_color(method), lw=1.8, label="Analysis"),
        Line2D([], [], color=S.INK, marker="*", ms=10, ls="none", mec=S.SURFACE,
               label="Release point (numbered)"),
    ]
