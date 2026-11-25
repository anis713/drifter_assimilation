"""Configuration, device selection and sanity checks for the assimilation notebook.

Nothing here touches the assimilation itself: it only describes a run and
checks that the requested run is one the code in ``src/`` can do.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SECONDS_PER_DAY = 86400.0
RAW_DIR_ENV = "DRIFTERS_RAW_DIR"   # environment variable naming the raw-download folder


@dataclass
class Config:
    # --- paths -----------------------------------------------------------
    root: Path
    data_dir: Path
    output_dir: Path
    truth_file: Path
    back_file: Path
    exp_name: str
    raw_truth: Path | None = None   # optional raw CMEMS reanalysis (truth)
    raw_back: Path | None = None    # optional raw altimetry (background)

    # --- domain and period ----------------------------------------------
    lon: list = field(default_factory=lambda: [30.0, 32.0])
    lat: list = field(default_factory=lambda: [33.0, 35.0])
    start_day: int = 0
    n_days: int = 3                 # assimilation length; n_days + 1 daily fields

    # --- drifters --------------------------------------------------------
    lon0: list = field(default_factory=list)
    lat0: list = field(default_factory=list)

    # --- time stepping ---------------------------------------------------
    dt_advection: float = 1800.0
    dt_observation: float = 1800.0

    # --- cost function and optimiser ------------------------------------
    lfil: int = 15000               # filter length scale (m)
    mylr: float = 0.01              # Adam learning rate
    alpha: float = 0.001            # weight of ||dU||^2
    beta: float = 0.1               # weight of the divergence penalty
    toll: float = 1.0               # stop when the gradient norm changes < toll %

    # --- method-specific -------------------------------------------------
    n_corr: int = 3                 # standard: number of constant-dU sub-windows
    tw_days: int = 1                # sliding: window length (days)
    sigma_days: int = 1             # sliding: window shift (days)

    # --- run control -----------------------------------------------------
    methods: tuple = ("standard", "sliding")
    rerun: bool = False
    device: str = "auto"            # "auto", "cpu" or "cuda"
    verbose: bool = False           # False: src/ progress output goes to a log file

    # --- figures and movies ---------------------------------------------
    frame_every: int = 2            # sliding movie: one frame every k advection steps
    fps: int = 8
    save_figures: bool = True
    save_movies: bool = True
    movie_format: str = "mp4"       # "mp4" (needs ffmpeg) or "gif"

    def __post_init__(self):
        for name in ("root", "data_dir", "output_dir", "truth_file", "back_file"):
            setattr(self, name, Path(getattr(self, name)))
        for name in ("raw_truth", "raw_back"):
            if getattr(self, name) is not None:
                setattr(self, name, Path(getattr(self, name)))
        self.methods = tuple(self.methods)

    # --- derived quantities ---------------------------------------------
    @property
    def n_drif(self) -> int:
        return len(self.lon0)

    @property
    def steps_per_day(self) -> int:
        return int(round(SECONDS_PER_DAY / self.dt_advection))

    @property
    def end_day(self) -> int:
        """Exclusive end index of the daily fields to load."""
        return self.start_day + self.n_days + 1

    @property
    def h5_path(self) -> Path:
        return self.output_dir / f"{self.exp_name}.h5"

    @property
    def fig_dir(self) -> Path:
        return self.output_dir / "figures" / self.exp_name

    def fingerprint(self) -> str:
        """Hash of every setting that changes assimilation results.

        Results stored under a different fingerprint are recomputed, so a
        stale HDF5 file is never silently re-plotted.
        """
        keys = ("truth_file", "back_file", "lon", "lat", "start_day", "n_days",
                "lon0", "lat0", "dt_advection", "dt_observation", "lfil", "mylr",
                "alpha", "beta", "toll", "n_corr", "tw_days", "sigma_days")
        d = asdict(self)
        payload = {k: (str(d[k]) if isinstance(d[k], Path) else d[k]) for k in keys}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def add_src_to_path(root: Path) -> Path:
    """Make the sibling modules of ``src/`` importable (they import each other flat)."""
    src = Path(root) / "src"
    if not (src / "adv_torch.py").exists():
        raise FileNotFoundError(
            f"{src} does not contain adv_torch.py. Run the notebook from the "
            "repository's notebooks/ folder, or fix ROOT in the config cell."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def select_device(preference: str = "auto"):
    """CUDA when available, otherwise CPU.

    Apple MPS is deliberately not used: the advection loop is thousands of
    tiny sequential kernels, where MPS is slower than the CPU.
    """
    import torch

    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE='cuda' was requested but CUDA is not available.")
    return torch.device(preference)


def check_config(cfg: Config) -> None:
    """Fail early on settings the ``src/`` code cannot handle."""
    problems = []
    if cfg.n_drif == 0:
        problems.append("LON0/LAT0 are empty: give at least one release point.")
    if len(cfg.lon0) != len(cfg.lat0):
        problems.append(f"LON0 has {len(cfg.lon0)} entries but LAT0 has {len(cfg.lat0)}.")
    if SECONDS_PER_DAY % cfg.dt_advection or SECONDS_PER_DAY % cfg.dt_observation:
        problems.append("dt_advection and dt_observation must divide one day (86400 s).")
    if cfg.dt_observation != cfg.dt_advection:
        # run_onTruth_new stores synthetic observations at every advection step,
        # and both assimilation functions assume that layout.
        problems.append("dt_observation must equal dt_advection for synthetic observations.")
    if "standard" in cfg.methods and cfg.n_days % cfg.n_corr:
        # multi_analysis_optm slices the last window wrongly otherwise.
        problems.append(f"n_days ({cfg.n_days}) must be divisible by n_corr ({cfg.n_corr}).")
    if "sliding" in cfg.methods:
        if not 1 <= cfg.tw_days <= cfg.n_days:
            problems.append(f"tw_days must be between 1 and n_days ({cfg.n_days}).")
        if cfg.sigma_days < 1:
            problems.append("sigma_days must be at least 1.")
    unknown = set(cfg.methods) - {"standard", "sliding"}
    if unknown:
        problems.append(f"Unknown method(s) {sorted(unknown)}; use 'standard' and/or 'sliding'.")
    for lo, la in zip(cfg.lon0, cfg.lat0):
        if not (cfg.lon[0] < lo < cfg.lon[1] and cfg.lat[0] < la < cfg.lat[1]):
            problems.append(f"Release point ({lo}, {la}) is outside LON={cfg.lon}, LAT={cfg.lat}.")
    problems += _input_problems(cfg)
    if problems:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(problems))


def _show(path: Path, root: Path) -> str:
    """``path`` relative to the repository when it lies inside it."""
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def _input_problems(cfg: Config) -> list[str]:
    """The processed inputs must exist, or both raw files must, for §2b to build them."""
    missing = [f for f in (cfg.truth_file, cfg.back_file) if not f.exists()]
    if not missing:
        return []
    names = ", ".join(_show(f, cfg.root) for f in missing)
    hint = f"Set {RAW_DIR_ENV} to the folder holding the raw CMEMS downloads (default: data/raw)."
    if cfg.raw_truth is None or cfg.raw_back is None:
        return [f"Missing input file(s): {names}. Set raw_truth and raw_back in the "
                f"configuration cell so they can be built. {hint}"]
    problems = []
    for raw_dir in sorted({cfg.raw_truth.parent, cfg.raw_back.parent}):
        if not raw_dir.is_dir():
            problems.append(f"Missing input file(s): {names}, and the raw-data folder "
                            f"{_show(raw_dir, cfg.root)} does not exist. {hint}")
    if problems:
        return problems
    for raw in (cfg.raw_truth, cfg.raw_back):
        if not raw.exists():
            problems.append(f"Raw file {raw.name} not found in {_show(raw.parent, cfg.root)} "
                            f"(needed to build {names}). Put it there, or {hint[0].lower()}{hint[1:]}")
    return problems


def check_release_points(cfg: Config, lon, lat, mask, margin: int = 2) -> None:
    """Each release point must be over water and ``margin`` cells from the edge.

    ``simul_drifter`` stops recording when a drifter leaves the domain, so
    points near the boundary give truncated trajectories.
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    problems = []
    for n, (lo, la) in enumerate(zip(cfg.lon0, cfg.lat0), start=1):
        i = int(np.abs(lon - lo).argmin())
        j = int(np.abs(lat - la).argmin())
        if not (margin <= i < lon.size - margin and margin <= j < lat.size - margin):
            problems.append(f"drifter {n} ({lo}, {la}) is within {margin} grid cells of the edge")
        elif mask[i, j]:
            problems.append(f"drifter {n} ({lo}, {la}) is on land")
    if problems:
        raise ValueError("Move these release points:\n  - " + "\n  - ".join(problems))


@contextlib.contextmanager
def src_output(cfg: Config, name: str):
    """Send the per-iteration prints and progress bars of ``src/`` to a log file.

    With ``cfg.verbose`` the output stays in the notebook.
    """
    if cfg.verbose:
        yield None
        return
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output_dir / f"{cfg.exp_name}_{name}.log"
    with open(path, "w") as fh, contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
        yield path
