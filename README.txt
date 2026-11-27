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
  ├── ml/                          Learning optimal release locations
  │                                (see MACHINE LEARNING below)
  ├── output/                      Results, logs, figures, movies (generated)
  └── src/
      ├── analysis_nDrif.py        Preprocessing, synthetic observations,
      │                            standard assimilation
      ├── sliding_window_assim.py  Sliding-window assimilation
      ├── adv_torch.py             RK4 advection, smoothing filter,
      │                            divergence penalty
      ├── manifolds.py             Fixed points and unstable manifolds
      │                            (baseline release strategy)
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
  MACHINE LEARNING: OPTIMAL DRIFTER RELEASE LOCATIONS
--------------------------------------------------------------------------------

Code for "Learning Optimal Drifter Release Locations for Surface Current
Estimation in the Levantine Mediterranean" (Hammoud, Brajard and Issa).

A U-Net is trained to predict, from the first-day background (altimetry) and
true (reanalysis) velocities on a 16 x 16 grid, the gain map: the relative
RMSE reduction (in %) obtained by assimilating a single drifter released at
each point of the inner 10 x 10 grid. The predicted maps guide where drifters
are released, and the result is compared with releases near the unstable
manifold of the flow.

Scripts in ml/, in the order they are run (all settings are in ml/config.py):

  01_training_size.py         Validation/test loss vs. number of training
                              years
  02_cv_architecture.py       Blocked 5-fold cross-validation of the
                              architecture (--experiment depth | width_dropout)
  03_train_ensemble.py        Trains the 20 final models (fixed 26 epochs,
                              seeds 0-19) and the ensemble-size curve
  04_evaluate_ensemble.py     12-member bagged prediction on the test set,
                              regression and classification metrics
                              (--region A | B)
  05_deployment_experiments.py
                              Drifter releases with the assimilation of src/:
                              50 random releases in the CNN-predicted positive
                              region and near the unstable manifold, and single
                              releases at the predicted maximum and at the
                              manifold point of smallest det(J)
                              (--region A | B, --sample N | --all)
  06_summarize_deployments.py Figures 6-9 and the mu_t scores
                              (--region A | B, --release random | deterministic)
  fig_accuracy.py             Figure 4   (test-set classification accuracy)
  fig_overlays.py             Figure 5   (true vs. predicted classes)
  fig_mahalanobis.py          Figure 10  (input distribution shift)
  fig_vorticity.py            Figure 11  (gain vs. reanalysis vorticity)

Helper modules: config.py, data.py, model.py, training.py, ensemble.py,
plotting.py.

Example (from the repository root):

  python ml/03_train_ensemble.py
  python ml/04_evaluate_ensemble.py --region A
  python ml/05_deployment_experiments.py --region A --all
  python ml/06_summarize_deployments.py --region A --release random

Data. The gain-map datasets are HDF5 files with one group per weekly sample
(map_YYYY-MM-DD_to_YYYY-MM-DD/io_map/ with Ub, Vb, Ut, Vt of shape
(7, 16, 16) indexed (day, lon, lat), lon, lat and rdrmse of shape (10, 10)).
They are expected in data/ml/:

  region_A.h5               Region A, weekly samples from 1993 (training,
                            validation and test)
  region_B.h5               Region B, inputs only (generalisation, no
                            retraining)
  region_B_test_truth.h5    True gain maps of the 160 Region B test samples

They will be archived with the trained models on Zenodo. The code that
generates them (one assimilation per grid point and week, run on an HPC
cluster) will be added to this repository.

Split (53 weekly samples per year block): validation = blocks 14 and 18,
test = from block 28 on (Region A: 11 Nov 2022 - 31 Jul 2025, 133 samples),
training = blocks 4-27 except the validation blocks.

Environment variables:

  DRIFTERS_ML_DATA      data folder              (default: data/ml)
  DRIFTERS_ML_OUTPUT    results folder           (default: output/ml)
  DRIFTERS_ML_WORKERS   parallel workers for 01/02

Training uses the CPU; the 05 experiments take several hours per region
because each release runs a full sliding-window assimilation.


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
