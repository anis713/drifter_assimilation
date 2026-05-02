import io
from analysis_nDrif import check_if_exists
import matplotlib.pyplot as plt
import numpy as np
from adv_torch import avgVfield_np, read_nc
import cartopy.crs as ccrs
def display_window_new(iwindow, LON, LAT, Ut, Vt, Ub, Vb, lon, lat, assim_grp, nDaysSim, n_drif,
                   fil, alpha, beta, tol, lr):
    Ut = Ut.detach().numpy()
    Vt = Vt.detach().numpy()
    Ub = Ub.detach().numpy()
    Vb = Vb.detach().numpy()
    dU = assim_grp[f"dU/dUincrement_{iwindow}window_{nDaysSim}days_fil{fil}_alp{alpha:.6f}_bet{beta:.6f}_toll{tol:.2f}_lr{lr:.6f}"][:]
    dV = assim_grp[f"dU/dVincrement_{iwindow}window_{nDaysSim}days_fil{fil}_alp{alpha:.6f}_bet{beta:.6f}_toll{tol:.2f}_lr{lr:.6f}"][:]

    avgUt, avgVt = avgVfield_np(Ut, Vt, nDaysSim)
    avgUb, avgVb = avgVfield_np(Ub, Vb, nDaysSim)
    avgUc, avgVc = avgVfield_np(Ub+dU, Vb+dV, nDaysSim)

    x, y = lon, lat
    lon_min, lon_max= LON[0], LON[1]
    lat_min, lat_max= LAT[0], LAT[1]

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(11,8.5))
    #ax.add_feature(cartopy.feature.OCEAN, zorder=0)
    #ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')
    ax.coastlines('50m')
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    #extet: lon_min, lon_max, lat_min, lat_max
    scale = 4
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
    ax.quiver(x,y,avgUb,avgVb, transform=ccrs.PlateCarree(), scale=scale, color='b', alpha=0.6, label='background')
    ax.quiver(x,y,avgUc,avgVc, transform=ccrs.PlateCarree(), scale=scale, color='m', alpha=0.6, label='analysis')
    ax.quiver(x,y,avgUt,avgVt, transform=ccrs.PlateCarree(), scale=scale, color='k', alpha=0.6, label='truth')

    # Read trajectories
    for nd in range(1, n_drif+1):
        lon_truth = assim_grp[f"metadata/lon_obs0{nd}_{nDaysSim}days"][:].reshape(-1)
        lat_truth = assim_grp[f"metadata/lat_obs0{nd}_{nDaysSim}days"][:].reshape(-1)

        lon_back = assim_grp[f"traj_back/lon_traj0{nd}_back{iwindow}window_{nDaysSim}days_fil{fil}_alp{alpha:.6f}_bet{beta:.6f}_toll{tol:.2f}_lr{lr:.6f}"][:].reshape(-1)
        lat_back = assim_grp[f"traj_back/lat_traj0{nd}_back{iwindow}window_{nDaysSim}days_fil{fil}_alp{alpha:.6f}_bet{beta:.6f}_toll{tol:.2f}_lr{lr:.6f}"][:].reshape(-1)

        lon_corr = assim_grp[f"traj_last/lon_traj0{nd}_last{iwindow}window_{nDaysSim}days_fil{fil}_alp{alpha:.6f}_bet{beta:.6f}_toll{tol:.2f}_lr{lr:.6f}"][:].reshape(-1)
        lat_corr = assim_grp[f"traj_last/lat_traj0{nd}_last{iwindow}window_{nDaysSim}days_fil{fil}_alp{alpha:.6f}_bet{beta:.6f}_toll{tol:.2f}_lr{lr:.6f}"][:].reshape(-1)

        
        ax.plot(lon_back, lat_back,  '-b', transform=ccrs.PlateCarree(), label=f'background traj')
        
        color = plt.get_cmap("tab10")((nd-1) % 10)
        ax.plot(lon_corr, lat_corr,  '-', color="m", transform=ccrs.PlateCarree(), label=f'analysis traj {nd}')
        
        ax.plot(lon_truth, lat_truth,  '-k', label=f'obs traj {nd}', transform=ccrs.PlateCarree())
        
        ax.plot(lon_corr[0], lat_corr[0],  '*r', transform=ccrs.PlateCarree())
    
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    fig_grp = assim_grp.require_group("figures")

    png_array = np.frombuffer(buf.getvalue(), dtype='uint8')
    fig_name = f"FIG_{iwindow}window_{nDaysSim}days_fil{fil}_alp{alpha:.6f}_bet{beta:.6f}_toll{tol:.2f}_lr{lr:.6f}"
    check_if_exists(fig_name, fig_grp)
    fig_grp.create_dataset(fig_name, data=png_array)

def display_window(iwindow, LON, LAT, input_dir, out_dir, saving_dir, nDaysSim, n_drif,
                   truth, back, corr, any_nc_folder,
                   fil, alpha, beta, tol, lr):
    # ROOT = Path(__file__).resolve().parents[-1]
    # out_dir = ROOT / "output"
    # Reading fields
    
    # Background
    Uback = np.load(back)
    Ub, Vb = Uback["Ut"], Uback["Vt"]

    avgUb, avgVb = avgVfield_np(Ub, Vb, nDaysSim)

    # Truth
    Utrue = np.load(truth)
    Ut, Vt = Utrue["Ut"], Utrue["Vt"]

    avgUt, avgVt = avgVfield_np(Ut, Vt, nDaysSim)

    # Corrected
    dUt = np.load(corr)
    du, dv = dUt["dUt"][0], dUt["dVt"][0]
    Uc, Vc = Ub + du, Vb + dv

    avgUc, avgVc = avgVfield_np(Uc, Vc, nDaysSim)

    # Read lon/lat
    filesList = sorted(any_nc_folder.glob("*.nc"))
    _, _, lon, lat = read_nc(filesList[0], LON, LAT)

    x, y = lon, lat
    lon_min, lon_max= LON[0], LON[1]
    lat_min, lat_max= LAT[0], LAT[1]

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(11,8.5))
    #ax.add_feature(cartopy.feature.OCEAN, zorder=0)
    #ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')
    ax.coastlines('50m')
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    #extet: lon_min, lon_max, lat_min, lat_max
    scale = 4
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
    ax.quiver(x,y,avgUb,avgVb, transform=ccrs.PlateCarree(), scale=scale, color='b', label='background')
    ax.quiver(x,y,avgUc,avgVc, transform=ccrs.PlateCarree(), scale=scale, color='m', label='analysis')
    ax.quiver(x,y,avgUt,avgVt, transform=ccrs.PlateCarree(), scale=scale, color='k', label='truth')

    # Read trajectories
    for nd in range(1, n_drif+1):
        obs_trj = input_dir / f"obs_{n_drif}drif_{nDaysSim}days" / f"obs0{nd}_{nDaysSim}days.npz"
        ontruth = np.load(obs_trj)
        lon_truth, lat_truth = ontruth["sim_lon"].reshape(-1), ontruth["sim_lat"].reshape(-1)

        back_trj = out_dir / f"traj0{nd}_back{iwindow}window_{nDaysSim}days_fil{fil}_alp{alpha:.6f}_bet{beta:.6f}_toll{tol:.2f}_lr{lr:.6f}.npz"
        onback = np.load(back_trj)
        lon_back, lat_back = onback["sim_lon"].reshape(-1), onback["sim_lat"].reshape(-1)

        corr_trj = out_dir / f"traj_last{nd}_{iwindow}window_{nDaysSim}days_fil{fil}_alp{alpha:.6f}_bet{beta:.6f}_toll{tol:.2f}_lr{lr:.6f}.npz"
        oncorr = np.load(corr_trj)
        lon_corr, lat_corr = oncorr["sim_lon"].reshape(-1), oncorr["sim_lat"].reshape(-1)

        ax.plot(lon_back, lat_back,  '-b', transform=ccrs.PlateCarree(), label=f'background traj')
        
        color = plt.get_cmap("tab10")((nd-1) % 10)
        ax.plot(lon_corr, lat_corr,  '-', color="m", transform=ccrs.PlateCarree(), label=f'analysis traj {nd}')
        
        ax.plot(lon_truth, lat_truth,  '-k', label=f'obs traj {nd}', transform=ccrs.PlateCarree())
        
        ax.plot(lon_corr[0], lat_corr[0],  '*r', transform=ccrs.PlateCarree())
    
    ax.legend()
    fig_name = saving_dir / f"FIG_{iwindow}window_{nDaysSim}days_fil{fil}_alp{alpha:.6f}_bet{beta:.6f}_toll{tol:.2f}_lr{lr:.6f}.png"
     
    fig.savefig(fig_name)

def display_loss(loss_path):
    lossarray = np.load(loss_path)
    loss, count = lossarray["obj"], lossarray["counts"]
    plt.plot(np.array(count), loss)
    plt.xlabel("iterations")
    plt.ylabel("loss")

def display_loss_new(assim_grp, iwindow, nDaysSim, lfil, alpha, beta, toll, mylr):
    loss = assim_grp[f"objective_function/obj_fct_behavior_window{iwindow}_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}"][:]
    plt.plot(loss)
    plt.xlabel("iterations")
    plt.ylabel("loss")

def display_rmse(mainh5, nDaysSim, lfil, alpha, beta, toll, mylr, Tw_days, sigma_days, METHOD):
    grp = mainh5[f"errors"]

    if METHOD == "standard":
        rms_b = grp[f"RMS_back_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}"][:]
        rms_c = grp[f"RMS_corr_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}"][:]
    elif METHOD == "sliding":
        rms_b = grp[f"RMS_back_Tw{Tw_days}d_sig{sigma_days}d_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}"][:]
        rms_c = grp[f"RMS_corr_Tw{Tw_days}d_sig{sigma_days}d_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}"][:]
    else:
        print("RMSE files not found.")

    end = len(rms_c)//48

    plt.plot(rms_b, color="b", label="RMSE(back)")
    plt.plot(rms_c, color="m", label="RMSE(corr)")
    plt.xticks([i*48 for i in range(end+1)], [f"Day{i}" for i in range(1,end+2)])
    plt.xlabel("Time (days)")
    plt.ylabel("%")
    plt.legend()
        






    

