"""Animations of velocity fields and drifter trajectories.

Both movies share one layout: three map panels (background | analysis | truth)
with speed shading and arrows, the drifter tracks on top, a strip with the
correction windows and a strip with ΔRMSE(t).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec

from . import style as S
from .plots import nanmean0, trajectory_legend_handles


# --------------------------------------------------------------------------- #
# Shared layout
# --------------------------------------------------------------------------- #
class _Triptych:
    def __init__(self, cfg, method, lon, lat, vmax, spans, rms, land):
        self.cfg, self.method = cfg, method
        self.lon, self.lat = np.asarray(lon), np.asarray(lat)
        self.vmax = vmax
        self.step = S.quiver_step(self.lon, self.lat)
        self.color = S.method_color(method)

        self.fig = plt.figure(figsize=(13.5, 6.9), layout="constrained")
        gs = GridSpec(3, 3, figure=self.fig, height_ratios=[1.0, 0.085, 0.27])
        titles = ["Background", f"Analysis · {S.METHOD_SHORT[method]}", "Truth"]
        self.maps, self.meshes, self.quivers = [], [], []
        nan = np.full((self.lon.size, self.lat.size), np.nan)
        for i, title in enumerate(titles):
            ax = S.add_map_axes(self.fig, gs[0, i], cfg.lon, cfg.lat, left_labels=(i == 0))
            self.meshes.append(S.speed_mesh(ax, self.lon, self.lat, nan, nan, vmax))
            S.land_mesh(ax, self.lon, self.lat, land)
            self.quivers.append(S.velocity_quiver(ax, self.lon, self.lat,
                                                  np.zeros_like(nan), np.zeros_like(nan),
                                                  vmax, self.step))
            S.label_releases(ax, cfg.lon0, cfg.lat0)
            S.panel_tag(ax, f"{'abc'[i]}  {title}")
            self.maps.append(ax)
        ref = S.nice_speed(vmax)
        self.maps[-1].quiverkey(self.quivers[-1], 0.80, 1.035, ref, f"{ref:g} m s$^{{-1}}$",
                                labelpos="E", coordinates="axes", color=S.INK,
                                labelcolor=S.INK_2, fontproperties={"size": 8})
        S.speed_colorbar(self.fig, self.meshes[0], self.maps)

        # correction-window strip
        self.ax_win = self.fig.add_subplot(gs[1, :])
        self.win_bars = []
        layers = 1
        if method == "sliding" and cfg.tw_days > cfg.sigma_days:
            layers = -(-cfg.tw_days // cfg.sigma_days)
        h = 1.0 / layers
        for k, (a, b) in enumerate(spans):
            y0 = (layers - 1 - k % layers) * h
            bar = self.ax_win.broken_barh([(a + 0.01, (b - a) - 0.02)], (y0 + 0.1 * h, 0.8 * h),
                                          facecolor=self.color, edgecolor="none", alpha=0.25)
            self.win_bars.append(bar)
        self.ax_win.set_xlim(0, cfg.n_days)
        self.ax_win.set_ylim(0, 1)
        self.ax_win.set_yticks([0.5], ["windows"], fontsize=8.5)
        self.ax_win.tick_params(labelbottom=False)
        self.ax_win.grid(False)
        for s in ("left", "bottom"):
            self.ax_win.spines[s].set_visible(False)

        # ΔRMSE strip
        self.ax_d = self.fig.add_subplot(gs[2, :], sharex=self.ax_win)
        self.ax_d.plot(rms["t_days"], rms["drms"], color=self.color, lw=1.8)
        self.ax_d.axhline(0, color=S.MUTED, lw=0.8)
        self.ax_d.set_ylabel("RMSE\nreduction (%)")
        self.ax_d.set_xlabel("Time since start of assimilation (days)")
        self.ax_d.set_xlim(0, cfg.n_days)

        self.fig.legend(handles=trajectory_legend_handles(method), loc="outside lower center",
                        ncols=4, fontsize=9)
        self.title = self.fig.suptitle("", x=0.01, ha="left", fontsize=12,
                                       fontweight="bold", color=S.INK)

    def set_fields(self, fields):
        """fields: three (U, V) pairs, arrays (nlon, nlat)."""
        st = self.step
        for mesh, q, (U, V) in zip(self.meshes, self.quivers, fields):
            mesh.set_array(np.ma.masked_invalid(np.hypot(U, V).T))
            q.set_UVC(np.ma.masked_invalid(U[::st, ::st].T), np.ma.masked_invalid(V[::st, ::st].T))

    def track(self, ax, color, lw=1.8, zorder=6):
        line, = ax.plot([], [], color=color, lw=lw, zorder=zorder,
                        solid_capstyle="round", **S.geo(ax))
        head, = ax.plot([], [], "o", ms=5.5, color=color, mec=S.SURFACE, mew=1.2,
                        zorder=zorder + 1, **S.geo(ax))
        return line, head

    def freeze_layout(self):
        """Fix the layout after the first draw so frames don't jitter."""
        self.fig.canvas.draw()
        self.fig.set_layout_engine("none")


def _set_track(line, head, x, y, alpha=1.0, show_head=True):
    line.set_data(x, y)
    line.set_alpha(alpha)
    ok = np.flatnonzero(np.isfinite(x))
    if show_head and ok.size:
        head.set_data([x[ok[-1]]], [y[ok[-1]]])
        head.set_alpha(alpha)
    else:
        head.set_data([], [])


def _vmax(*pairs):
    s = [np.hypot(U, V) for U, V in pairs]
    return float(np.nanpercentile(np.concatenate([a.ravel() for a in s]), 99.5))


# --------------------------------------------------------------------------- #
# Movie A: constant correction per sub-window (one frame per sub-window)
# --------------------------------------------------------------------------- #
def movie_standard_windows(cfg, lon, lat, Ub, Vb, Ut, Vt, dU_list, spans,
                           obs, traj_back, traj_last, rms):
    """One frame per sub-window: window-mean fields and that window's tracks.

    Ub, Vb, Ut, Vt: daily arrays (n_days + 1, nlon, nlat).
    dU_list: [(dU, dV)] per window (unfiltered, as stored by src/).
    obs: (lon, lat) arrays (n_drif, n_days * spd + 1), from results.load_obs.
    traj_back, traj_last: lists per window, from results.load_window_traj.
    """
    spd = cfg.steps_per_day
    K = len(dU_list)
    means = []
    for w, (a, b) in enumerate(spans):
        sl = slice(int(a), int(b) + 1)
        ub, vb = nanmean0(Ub[sl]), nanmean0(Vb[sl])
        ut, vt = nanmean0(Ut[sl]), nanmean0(Vt[sl])
        du, dv = dU_list[w]
        means.append([(ub, vb), (ub + du, vb + dv), (ut, vt)])
    vmax = _vmax(*[p for m in means for p in m])

    T = _Triptych(cfg, "standard", lon, lat, vmax, spans, rms, np.isnan(Ut[0]))
    bands = [T.ax_d.axvspan(a, b, color=T.color, alpha=0.12, lw=0, visible=False)
             for a, b in spans]

    # artists[w][panel] = list over drifters of (obs line/head, own line/head)
    own_color = [S.BACKGROUND, T.color, None]
    own_track = [traj_back, traj_last, None]
    artists = []
    for w in range(K):
        per_panel = []
        for p, ax in enumerate(T.maps):
            items = []
            for nd in range(cfg.n_drif):
                o = T.track(ax, S.TRUTH, lw=2.0, zorder=6)
                own = T.track(ax, own_color[p], lw=1.8, zorder=7) if own_color[p] else None
                items.append((o, own))
            per_panel.append(items)
        artists.append(per_panel)

    def window_drms(w):
        a, b = spans[w]
        sel = (rms["t_days"] >= a) & (rms["t_days"] < b)
        return float(np.nanmean(rms["drms"][sel]))

    def update(w):
        T.set_fields(means[w])
        for k in range(K):
            visible = k <= w
            alpha = 1.0 if k == w else 0.28
            i0, i1 = int(spans[k][0] * spd), int(spans[k][1] * spd)
            for p in range(3):
                for nd, (o, own) in enumerate(artists[k][p]):
                    if not visible:
                        _set_track(*o, [], [], show_head=False)
                        if own:
                            _set_track(*own, [], [], show_head=False)
                        continue
                    _set_track(*o, obs[0][nd, i0:i1 + 1], obs[1][nd, i0:i1 + 1],
                               alpha, show_head=(k == w))
                    if own:
                        tr = own_track[p][k]
                        _set_track(*own, tr[0][nd], tr[1][nd], alpha, show_head=(k == w))
        for k, (bar, band) in enumerate(zip(T.win_bars, bands)):
            bar.set_alpha(1.0 if k == w else 0.25)
            band.set_visible(k == w)
        a, b = spans[w]
        T.title.set_text(f"Constant correction per sub-window  ·  window {w + 1}/{K}, "
                         f"days {a:g}–{b:g}  ·  window-mean fields  ·  "
                         f"RMSE reduction {window_drms(w):.1f} %")
        return []

    update(0)
    T.freeze_layout()
    return animation.FuncAnimation(T.fig, update, frames=range(K),
                                   interval=1000 * max(1.0, 8 / cfg.fps), blit=False)


# --------------------------------------------------------------------------- #
# Movie B: sliding window (time-continuous)
# --------------------------------------------------------------------------- #
def movie_sliding(cfg, lon, lat, Ub_s, Vb_s, Uc_s, Vc_s, Ut_s, Vt_s, spans,
                  obs, back_full, anal_full, rms):
    """Instantaneous fields and growing tracks at every ``cfg.frame_every`` steps.

    *_s: fields at advection resolution (n_days * spd, nlon, nlat).
    obs, back_full, anal_full: (lon, lat) arrays (n_drif, n_days * spd + 1);
    anal_full is advected on daily samples of the blended field (approximation).
    """
    spd = cfg.steps_per_day
    n_steps = Uc_s.shape[0]
    frames = list(range(0, n_steps, cfg.frame_every))
    if frames[-1] != n_steps - 1:
        frames.append(n_steps - 1)
    vmax = _vmax((Ub_s, Vb_s), (Uc_s, Vc_s), (Ut_s, Vt_s))

    T = _Triptych(cfg, "sliding", lon, lat, vmax, spans, rms, np.isnan(Ut_s[0]))
    cursor = T.ax_d.axvline(0, color=S.INK, lw=1.0)
    head_d, = T.ax_d.plot([], [], "o", ms=5, color=T.color, mec=S.SURFACE, mew=1.2, zorder=5)

    own = [(S.BACKGROUND, back_full), (T.color, anal_full), (None, None)]
    tracks = []
    for p, ax in enumerate(T.maps):
        items = []
        for nd in range(cfg.n_drif):
            o_full = T.track(ax, S.TRUTH, lw=1.0, zorder=5)    # faint full future path
            o = T.track(ax, S.TRUTH, lw=2.0, zorder=6)
            c, _ = own[p]
            ow = T.track(ax, c, lw=1.8, zorder=7) if c else None
            items.append((o_full, o, ow))
        tracks.append(items)
    for p in range(3):
        for nd, (o_full, _, _) in enumerate(tracks[p]):
            _set_track(*o_full, obs[0][nd], obs[1][nd], alpha=0.18, show_head=False)

    def update(t):
        T.set_fields([(Ub_s[t], Vb_s[t]), (Uc_s[t], Vc_s[t]), (Ut_s[t], Vt_s[t])])
        for p in range(3):
            for nd, (_, o, ow) in enumerate(tracks[p]):
                _set_track(*o, obs[0][nd, :t + 1], obs[1][nd, :t + 1])
                if ow:
                    tr = own[p][1]
                    _set_track(*ow, tr[0][nd, :t + 1], tr[1][nd, :t + 1])
        td = t / spd
        for (a, b), bar in zip(spans, T.win_bars):
            bar.set_alpha(1.0 if a <= td <= b else 0.25)
        cursor.set_xdata([td, td])
        head_d.set_data([rms["t_days"][t]], [rms["drms"][t]])
        active = sum(a <= td <= b for a, b in spans)
        T.title.set_text(f"Sliding-window assimilation  ·  t = {td:5.2f} days  ·  "
                         f"{active} active window{'s' if active != 1 else ''}  ·  "
                         f"RMSE reduction {rms['drms'][t]:5.1f} %")
        return []

    update(frames[0])
    T.freeze_layout()
    return animation.FuncAnimation(T.fig, update, frames=frames,
                                   interval=1000 / cfg.fps, blit=False)


# --------------------------------------------------------------------------- #
# Display and export
# --------------------------------------------------------------------------- #
def show(anim, dpi=80):
    """Inline, scrubbable player (no ffmpeg needed).

    Rendered at a lower dpi as JPEG frames to keep the notebook light; the
    files written by ``save`` are full resolution.
    """
    from IPython.display import HTML
    import matplotlib as mpl
    with mpl.rc_context({"animation.frame_format": "jpeg", "savefig.dpi": dpi}):
        anim._fig.set_dpi(dpi)
        html = anim.to_jshtml(default_mode="once")
    plt.close(anim._fig)
    return HTML(html)


def save(anim, cfg, name: str, fps: float | None = None):
    """MP4 when ffmpeg is available, GIF otherwise. Returns the path or None."""
    if not cfg.save_movies:
        return None
    fps = fps or cfg.fps
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    if cfg.movie_format == "mp4" and animation.writers.is_available("ffmpeg"):
        path = cfg.fig_dir / f"{name}.mp4"
        writer = animation.FFMpegWriter(fps=fps, bitrate=3000,
                                        extra_args=["-pix_fmt", "yuv420p"])
    else:
        path = cfg.fig_dir / f"{name}.gif"
        writer = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=150)
    return path
