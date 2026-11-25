================================================================================
  Drifter-Based Ocean Current Assimilation
  Based on Issa et al. (2016)
================================================================================

Implementation of the data-assimilation framework described in Issa et al.
(2016) for correcting ocean-current background fields using Lagrangian drifter
observations.


--------------------------------------------------------------------------------
  OVERVIEW
--------------------------------------------------------------------------------

The method optimises a correction field dU that, when added to a background
velocity field Ub (e.g. from satellite altimetry), minimises the mismatch
between simulated and observed drifter trajectories.

Two assimilation strategies are provided, and the notebook runs and compares
both:

  standard  (multi_analysis_optm)
      dU is a constant correction field within each of n_corr sequential
      sub-windows.

  sliding   (multi_analysis_sliding_window)
      Corrections fitted on overlapping windows of Tw_days, shifted by
      sigma_days, are blended into a time-continuous dU(t) with weights
      1/(|k-k*|+1) (Eq. 5 of Issa et al. 2016).

Choose which ones to run with `methods` in the configuration cell (Section 1)
of the notebook.


--------------------------------------------------------------------------------
  REPOSITORY STRUCTURE
--------------------------------------------------------------------------------

  project_root/
  ├── data/
  │   ├── back_stand.nc            Background field (altimetry), July 2006
  │   └── truth_inter.nc           Truth field (reanalysis interpolated onto
  │                                the altimetry grid), July 2006
  ├── notebooks/
  │   ├── assimilation.ipynb       Main notebook  <- start here
  │   └── assim_nb/                Configuration, result readers, figures and
  │                                movies (no assimilation code)
  ├── output/                      Results, logs, figures, movies (generated)
  └── src/
      ├── analysis_nDrif.py        Preprocessing, synthetic observations,
      │                            standard assimilation
      ├── sliding_window_assim.py  Sliding-window assimilation
      ├── adv_torch.py             RK4 advection, smoothing filter,
      │                            divergence penalty
      └── multilinear.py           Differentiable bilinear interpolation


--------------------------------------------------------------------------------
  QUICK START
--------------------------------------------------------------------------------

1. Clone the repository

     git clone https://github.com/anis713/drifter_assimilation
     cd drifter_assimilation

2. Install dependencies

     pip install -r requirements.txt

   MP4 movies need ffmpeg (e.g. `brew install ffmpeg`); without it the
   notebook writes GIFs.

3. Open the notebook from the notebooks/ folder and run all cells

     cd notebooks/
     jupyter notebook assimilation.ipynb

   Or run it headless from the repository root:

     jupyter nbconvert --to notebook --execute notebooks/assimilation.ipynb \
         --output-dir <folder>

The input files in data/ are included, so the default experiment (two
drifters, 3 days, in the Levantine basin) runs without downloading anything.
Results go to output/notebooks/.


--------------------------------------------------------------------------------
  DATA
--------------------------------------------------------------------------------

The code was developed with the following Copernicus Marine Service (CMEMS)
products, freely available at:

    https://marine.copernicus.eu

  Background field
    Product : GLOBAL OCEAN GRIDDED L4 SEA SURFACE HEIGHTS AND DERIVED
              VARIABLES REPROCESSED (DUACS)
    ID      : SEALEVEL_GLO_PHY_L4_MY_008_047
    Variables: ugos, vgos  (geostrophic surface currents)
    Resolution: 0.125 deg, daily

  Truth / observation proxy
    Product : MEDITERRANEAN SEA PHYSICS REANALYSIS
    ID      : MEDSEA_MULTIYEAR_PHY_006_004
    Variables: uo, vo  (3-D ocean currents)
    Resolution: ~4.2 km, daily

The files in data/ are subsets of these products: daily surface currents for
1-31 July 2006 over 6W-36.5E, 30N-46N, on the 0.125 deg altimetry grid, with
variables ugos, vgos and dimensions (time, latitude, longitude). They are
distributed under the Copernicus Marine Service licence and were generated
using E.U. Copernicus Marine Service Information.

To use other dates or regions, download the two products and either replace
the files in data/, or delete them and set raw_truth / raw_back in the
configuration cell: the notebook then builds them from the raw downloads, read
from the folder in the environment variable DRIFTERS_RAW_DIR (default
data/raw/).

Other gridded NetCDF products should work provided they contain surface
zonal and meridional velocity components on a regular grid.


--------------------------------------------------------------------------------
  KEY ASSIMILATION PARAMETERS
--------------------------------------------------------------------------------

  lfil          Filter length scale                     [m]      default: 15000
  alpha         Weight on ||dU||^2 regularisation term           default: 0.001
  beta          Weight on divergence penalty                      default: 0.1
  mylr          Adam learning rate                                default: 0.01
  toll          Convergence tolerance on relative ||grad L||      default: 1.0
  n_corr        Number of sequential windows (standard only)      default: 3
  tw_days       Window width in days          (sliding only)      default: 1
  sigma_days    Window shift in days          (sliding only)      default: 1


--------------------------------------------------------------------------------
  REFERENCE
--------------------------------------------------------------------------------

Leila Issa, Julien Brajard, Milad Fakhri, Daniel Hayes, Laurent Mortier, and
Pierre-Marie Poulain. Modelling surface currents in the eastern Levantine
Mediterranean using surface drifters and satellite altimetry. Ocean Modelling,
104:1-14, 2016.


--------------------------------------------------------------------------------
  LICENSE
--------------------------------------------------------------------------------

MIT -- see LICENSE for details.

================================================================================
