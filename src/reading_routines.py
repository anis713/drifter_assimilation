import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from netCDF4 import Dataset

def scan_file(filename):
    dataset = Dataset(filename)
    for i in dataset.variables:
        print([i , dataset.variables[i].units , dataset.variables[i].shape])

def standardize_velocity(file_analysis, outfile=None):

    ds = xr.open_dataset(file_analysis, chunks={"time": 100})

    # rename variables
    rename_dict = {}
    if 'uo' in ds.data_vars: rename_dict['uo'] = 'ugos'
    if 'vo' in ds.data_vars: rename_dict['vo'] = 'vgos'
    ds = ds.rename(rename_dict)

    # keep velocity only
    ds = ds[['ugos','vgos']]

    # remove depth if present
    if "depth" in ds.dims:
        ds = ds.isel(depth=0, drop=True)

    # coordinate naming 
    rename_coords = {}
    if "lat" in ds.coords: rename_coords["lat"] = "latitude"
    if "lon" in ds.coords: rename_coords["lon"] = "longitude"
    ds = ds.rename(rename_coords)

    # reorder dimensions 
    ds = ds.transpose("time","latitude","longitude")

    # remove problematic fillvalue encodings
    for v in ds.data_vars:
        ds[v].encoding = {}

    encoding = {"ugos": {"_FillValue": np.nan},"vgos": {"_FillValue": np.nan}}

    if outfile is not None:
        ds.to_netcdf(outfile)

    return ds

def intrp_analysis_aviso1(rea_file, aviso_file, outfile):

    ds_rea = standardize_velocity(rea_file)
    ds_aviso = standardize_velocity(aviso_file)

    ds_regrid = ds_rea.interp(
        longitude=ds_aviso.longitude,
        latitude=ds_aviso.latitude
    )
    if outfile is not None:
        ds_regrid.to_netcdf(outfile)

    return ds_regrid


def read_daily_files(file_analysis, saving_dir, filetype):

    saving_dir = Path(saving_dir)
    saving_dir.mkdir(parents=True, exist_ok=True)
    file_analysis = Path(file_analysis)


    # --- 1. Open dataset ---
    ds = xr.open_dataset(file_analysis)

    # --- 2. Rename variables to match AVISO naming ---
    rename_dict = {}
    if 'uo' in ds.data_vars: rename_dict['uo'] = 'ugos'
    if 'vo' in ds.data_vars: rename_dict['vo'] = 'vgos'
    ds = ds.rename(rename_dict)

    # --- 3. Keep only velocity variables ---
    ds = ds[['ugos', 'vgos']]

    for t in ds.time.values:
        day = pd.to_datetime(t).strftime("%Y-%m-%d")

        # Extract slice
        ds_day = ds.sel(time=t)

        # --- FORCE TIME TO BECOME A DIMENSION ---
        if "time" not in ds_day.dims:
            ds_day = ds_day.expand_dims(time=[t])

        # Remove depth dimension if present
        if "depth" in ds_day.dims:
            ds_day = ds_day.isel(depth=0, drop=True)

        # Ensure coordinate names
        if "lat" in ds_day.coords:
            ds_day = ds_day.rename({"lat": "latitude"})
        if "lon" in ds_day.coords:
            ds_day = ds_day.rename({"lon": "longitude"})

        # Reorder dims safely
        dims = list(ds_day.dims)
        wanted = ["time", "latitude", "longitude"]
        dims_sorted = [d for d in wanted if d in dims]
        ds_day = ds_day.transpose(*dims_sorted)

        # ---- KEEP NaNs: remove conflicting attrs and disable FillValue encoding ----
        encoding = {}
        for v in ["ugos", "vgos"]:
            if v in ds_day:
                ds_day[v].attrs.pop("missing_value", None)
                ds_day[v].attrs.pop("_FillValue", None)
                encoding[v] = {"_FillValue": None, "dtype": "float32"}

        filename = saving_dir / f"{filetype}_velocity_{day}.nc"
        ds_day.to_netcdf(filename, encoding=encoding)
        print(f"Saved {filename}")

    print("This is a sample: ")

    first_file = next(p for p in saving_dir.iterdir() if p.is_file() and p.suffix.lower() == ".nc")

    print(first_file)
    print(xr.open_dataset(first_file))

def intrp_analysis_aviso(rea_dir, aviso_dir, out_dir):
    
    rea_dir = Path(rea_dir)
    aviso_dir = Path(aviso_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rea_files   = sorted([f for f in rea_dir.iterdir() if f.is_file() and f.suffix.lower() == ".nc"])
    aviso_files = sorted([f for f in aviso_dir.iterdir() if f.is_file() and f.suffix.lower() == ".nc"])

    for file_rea, file_aviso in zip(rea_files, aviso_files):

        print(f"\n→ Processing:\n   REA:   {file_rea}\n   AVISO: {file_aviso}")

        # --- Load datasets ---
        ds_rea = xr.open_dataset(file_rea)
        ds_aviso = xr.open_dataset(file_aviso)

        # --- Temporary rename to lat/lon for interpolation compatibility ---
        for old,new in [("latitude","lat"), ("longitude","lon")]:
            if old in ds_rea.coords:
                ds_rea = ds_rea.rename({old:new})
            if old in ds_aviso.coords:
                ds_aviso = ds_aviso.rename({old:new})

        # --- Match AVISO time with reanalysis time ---
        rea_time = ds_rea['time'].values
        ds_aviso = ds_aviso.sel(time=rea_time, method="nearest")

        # --- Interpolate reanalysis onto AVISO grid ---
        ds_regrid = ds_rea.interp(
            lon=ds_aviso['lon'],
            lat=ds_aviso['lat'],
            method="linear"
        )

        # Rename coordinates back safely
        rename_back = {}
        if "lat" in ds_regrid.coords: rename_back["lat"] = "latitude"
        if "lon" in ds_regrid.coords: rename_back["lon"] = "longitude"
        ds_regrid = ds_regrid.rename(rename_back)

        # --- Save output ---
        date_str = file_rea.name.split("velocity_")[-1].replace(".nc", "")
        out_path = out_dir / f"reanalysis_on_aviso_{date_str}.nc"
        ds_regrid.to_netcdf(out_path)

        print(f"   ✔ Saved: {out_path}")

