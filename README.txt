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
  │   ├── raw/
  │   │   ├── raw_data.zip         Raw CMEMS downloads, July 2006 (included,
  │   │   │                        unzip before the first run)
  │   │   ├── (new_aviso)cmems_obs-sl_glo_phy-ssh_...nc        background
  │   │   └── (new_reanalysis)cmems_mod_med_phy-cur_...nc      truth
  │   │                                                (both from raw_data.zip)
  │   ├── back_stand.nc            Standardised background   (generated)
  │   ├── truth_stand.nc           Standardised truth        (generated)
  │   └── truth_inter.nc           Truth interpolated onto the altimetry
  │                                grid                      (generated)
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

3. Unzip the raw data (once). The two raw .nc files are compressed in
   data/raw/raw_data.zip and must be extracted before running the notebook:

     unzip data/raw/raw_data.zip -d data/raw

   Without unzip:  python -m zipfile -e data/raw/raw_data.zip data/raw
   (or extract it with your file manager into data/raw/).

4. Open the notebook from the notebooks/ folder and run all cells

     cd notebooks/
     jupyter notebook assimilation.ipynb

   Or run it headless from the repository root:

     jupyter nbconvert --to notebook --execute notebooks/assimilation.ipynb \
         --output-dir <folder>

The raw input files are included in data/raw/raw_data.zip (37 MB, 107 MB
unzipped). Once they are unzipped, the first run of the notebook preprocesses
them (Section 2b) into the files in data/, so the default
experiment (two drifters, 3 days, in the Levantine basin) runs without
downloading anything. Results go to output/notebooks/.



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

The files in data/raw/raw_data.zip are subsets of these products, downloaded
unchanged:
daily surface currents for 1-31 July 2006 over the Mediterranean Sea
(6W-36.5E, 30N-46N). They are distributed under the Copernicus Marine Service
licence and were generated using E.U. Copernicus Marine Service Information.

The notebook's preprocessing (Section 2b) renames the velocities to ugos/vgos,
drops the depth axis and interpolates the reanalysis onto the 0.125 deg
altimetry grid. It runs only when the generated files in data/ are missing.

To use other dates or regions, download the two products into data/raw/ (or
into the folder named by the environment variable DRIFTERS_RAW_DIR), set
raw_truth / raw_back in the configuration cell, and delete the generated
files in data/ so they are rebuilt.

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
