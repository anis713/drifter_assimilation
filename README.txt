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

Two assimilation strategies are provided:

  standard  (multi_analysis_optm)
      dU is a constant correction field within each sequential analysis window.

  sliding   (multi_analysis_sliding_window)
      Overlapping Gaussian-weighted windows reconstruct a continuous dU(t)
      over the full assimilation period.

Choose your strategy by setting METHOD in Section 4 of the notebook:

    METHOD = "standard"   # or "sliding"


--------------------------------------------------------------------------------
  REPOSITORY STRUCTURE
--------------------------------------------------------------------------------

  project_root/
  ├── data/                        Input .nc files and preprocessed outputs
  │   ├── back_stand.nc            Standardised background field  (generated)
  │   ├── truth_stand.nc           Standardised truth field       (generated)
  │   └── truth_stand_inter.nc     Interpolated truth field       (generated)
  ├── output/                      Experiment results (.h5 files)
  └── src/
      ├── assim_nb.ipynb           Main notebook  ← start here
      ├── analysis_nDrif.py        Core engine: preprocessing, advection,
      │                            standard assimilation
      ├── sliding_window_assim.py  Sliding-window assimilation
      ├── reading_routines.py      NetCDF reading and scanning utilities
      └── plots.py                 Visualisation routines


--------------------------------------------------------------------------------
  QUICK START
--------------------------------------------------------------------------------

1. Clone the repository

     git clone https://github.com/<your-username>/<repo-name>.git
     cd <repo-name>

2. Install dependencies

     pip install -r requirements.txt

3. Download input data  (see DATA section below)

4. Place your .nc files in data/ and update the filenames in Section 2
   of the notebook:

     back_filename  = "your_background_file.nc"
     truth_filename = "your_truth_file.nc"

5. Open the notebook

     cd src/
     jupyter notebook assim_nb.ipynb

6. Run all cells in order.


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

Other gridded NetCDF products should work provided they contain surface
zonal and meridional velocity components on a regular grid.


--------------------------------------------------------------------------------
  KEY ASSIMILATION PARAMETERS
--------------------------------------------------------------------------------

  lfil          Gaussian filter half-width              [m]      default: 15000
  alpha         Weight on ||dU||^2 regularisation term           default: 0.001
  beta          Weight on divergence penalty                      default: 0.1
  mylr          Adam learning rate                                default: 0.01
  toll          Convergence tolerance on relative ||grad L||      default: 1.0
  n_corr        Number of sequential windows (standard only)      default: 3
  Tw_days       Window width in days          (sliding only)      default: 3
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
