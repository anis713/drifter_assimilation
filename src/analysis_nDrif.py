import os
import glob
import torch
import numpy as np
from adv_torch import simul_drifter, read_nc, get_mask, GenerateCoefs, ApplyFilter, diverg_penalty, timeinterpolate
from numpy import linalg as LA
from scipy.io import savemat
from decimal import Decimal, getcontext
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr

R_earth = 6371229
# torch.autograd.set_detect_anomaly(True)
def field_int_new(Ut, Vt, dtAdvection):
    # Ut,Vt are assumed CUDA tensors
    device = Ut.device
    dtype  = Ut.dtype
    nAdvectionStepPerDay = int(24.0*3600.0/dtAdvection)
    nDaysSim = Ut.shape[0]
    U_interpolated = torch.empty((nDaysSim-1, nAdvectionStepPerDay, *Ut[0].shape), device=device, dtype=dtype)
    V_interpolated = torch.empty((nDaysSim-1, nAdvectionStepPerDay, *Vt[0].shape), device=device, dtype=dtype)

    for iFilename in tqdm(range(0, nDaysSim-1)):
        if iFilename==0: #TODO: add a AVISO file for time=0
            Uday1, Vday1 = Ut[0], Vt[0]

        else:
            Uday1=Uday2
            Vday1=Vday2
        Uday2, Vday2 = Ut[iFilename+1], Vt[iFilename+1]

        for iAdvStepDay in range(0, nAdvectionStepPerDay):

            t0 = iAdvStepDay*dtAdvection
            
            U_interpolated[iFilename][iAdvStepDay], V_interpolated[iFilename][iAdvStepDay] = timeinterpolate(U1=Uday1, U2=Uday2,
                                                                                                    V1=Vday1, V2=Vday2, t=t0)
    return  U_interpolated, V_interpolated
def field_int(Ut, Vt, dtAdvection, saving_file):
    # Ut,Vt are assumed CUDA tensors
    device = Ut.device
    dtype  = Ut.dtype
    nAdvectionStepPerDay = int(24.0*3600.0/dtAdvection)
    nDaysSim = Ut.shape[0]
    U_interpolated = torch.empty((nDaysSim-1, nAdvectionStepPerDay, *Ut[0].shape), device=device, dtype=dtype)
    V_interpolated = torch.empty((nDaysSim-1, nAdvectionStepPerDay, *Vt[0].shape), device=device, dtype=dtype)

    for iFilename in tqdm(range(0, nDaysSim-1)):
        if iFilename==0: #TODO: add a AVISO file for time=0
            Uday1, Vday1 = Ut[0], Vt[0]

        else:
            Uday1=Uday2
            Vday1=Vday2
        Uday2, Vday2 = Ut[iFilename+1], Vt[iFilename+1]

        for iAdvStepDay in range(0, nAdvectionStepPerDay):

            t0 = iAdvStepDay*dtAdvection
            
            U_interpolated[iFilename][iAdvStepDay], V_interpolated[iFilename][iAdvStepDay] = timeinterpolate(U1=Uday1, U2=Uday2,
                                                                                                    V1=Vday1, V2=Vday2, t=t0)
    
    np.savez(saving_file, U_int=U_interpolated.detach().cpu().numpy(), V_int=V_interpolated.detach().cpu().numpy())
    return  U_interpolated, V_interpolated
                                                               
def RMS_err(field_dir, saving_dir, dU_path, window, t_w, total_days, dtAdvection, coefu, coefv):
    nAdvectionStepPerDay = int(24.0*3600.0/dtAdvection)
    
    filesList = glob.glob(os.path.join(field_dir, '*.nc'))
    filesList.sort()
    
    Ut1, Vt1, lon, lat = read_nc(filesList[window*t_w])
    Ut1, Vt1 = Ut1.detach().numpy(), Vt1.detach().numpy()
    
    Ut2, Vt2, lon, lat = read_nc(filesList[window*t_w+1])
    Ut2, Vt2 = Ut2.detach().numpy(), Vt2.detach().numpy()    

    back_dir1 = (saving_dir + '/background' + str(window+1) + '_'  
                + str(total_days) + 'days.npz')
    back1 = np.load(back_dir1)
    Ub1, Vb1 = back1['Ut'][0], back1['Vt'][0]
    if t_w == 1 and t_w*(window+1) == total_days:
        Ub2, Vb2 = coefu * Ut2, coefv * Vt2 
    else:   
        back_dir2 = (saving_dir + '/background' + str(window+2) + '_'  
                    + str(total_days) + 'days.npz')
        back2 = np.load(back_dir2)
        Ub2, Vb2 = back2['Ut'][0], back2['Vt'][0]

    grad = np.load(dU_path)
    dU = grad['dUt'][0]
    dV = grad['dVt'][0]

    rms_b = []
    rms_corr = []

    # for y in range(0,16):
    for iAdvStepDay in range(0, nAdvectionStepPerDay):
        t0 = iAdvStepDay*dtAdvection
        Ut_int = Ut1 + (Ut2 - Ut1) * t0 / (1 * 24 * 3600)
        Ub_int = Ub1 + (Ub2 - Ub1) * t0 / (1 * 24 * 3600)
    
        Vt_int = Vt1 + (Vt2 - Vt1) * t0 / (1 * 24 * 3600)
        Vb_int = Vb1 + (Vb2 - Vb1) * t0 / (1 * 24 * 3600)
        rms_b.append(100*np.sqrt(np.sum((Ut_int-Ub_int)**2 + (Vt_int-Vb_int)**2)/np.sum((Ut_int)**2 + (Vt_int)**2)))
        rms_corr.append(100*np.sqrt(np.sum((Ut_int-Ub_int-dU)**2 + (Vt_int-Vb_int-dV)**2)/np.sum((Ut_int)**2 + (Vt_int)**2)))
    rms_path = saving_dir + '/RMS_window' + str(window) + '.npz'
    np.savez(rms_path, rms_b=rms_b, rms_corr=rms_corr)
    return rms_b, rms_corr
    
        

def RMS(U_truth, V_truth, Ut_next, Vt_next, Ub, Vb, Ub_next, Vb_next, dU, dV, days, dtAdvection):
    nAdvectionStepPerDay = int(24.0*3600.0/dtAdvection)
    RMS = []
    if days == 1:
        U1, V1 = U_truth[0], V_truth[0]
        U2, V2 = Ut_next[0], Vt_next[0]
        
        Ub1, Vb1 = Ub[0], Vb[0]
        Ub2, Vb2 = Ub_next[0], Vb_next[0]
        
        for iAdvStepDay in range(0, nAdvectionStepPerDay):
            t0 = iAdvStepDay*dtAdvection
            # Interpolate true and back Field
            Ut_int = U1 + (U2 - U1) * t0 /  (1 * 24 * 3600)
            Vt_int = V1 + (V2 - V1) * t0 /  (1 * 24 * 3600)
            
            Ub_int = Ub1 + (Ub2 - Ub1) * t0 /  (1 * 24 * 3600)
            Vb_int = Vb1 + (Vb2 - Vb1) * t0 /  (1 * 24 * 3600)
            # Compute relative RMS error at this step
            num = (Ut_int - Ub_int - dU)**2 + (Vt_int - Vb_int - dV)**2
            den = Ut_int**2 + Vt_int**2

            err = torch.sqrt(torch.sum(num) / (torch.sum(den)))
            RMS.append(err.item())
    else:
        for iday in range(days-1):
            U1, V1 = U_truth[iday], V_truth[iday]
            U2, V2 = U_truth[iday+1], V_truth[iday+1]
            
            Ub1, Vb1 = Ub[iday], Vb[iday]
            Ub2, Vb2 = Ub[iday+1], Vb[iday+1]
            for iAdvStepDay in range(0, nAdvectionStepPerDay):
                t0 = iAdvStepDay*dtAdvection
                # Interpolate true and back Field
                Ut_int = U1 + (U2 - U1) * t0 /  (1 * 24 * 3600)
                Vt_int = V1 + (V2 - V1) * t0 /  (1 * 24 * 3600)
                
                Ub_int = Ub1 + (Ub2 - Ub1) * t0 /  (1 * 24 * 3600)
                Vb_int = Vb1 + (Vb2 - Vb1) * t0 /  (1 * 24 * 3600)
                
                # Compute relative RMS error at this step
                num = (Ut_int - Ub_int - dU)**2 + (Vt_int - Vb_int - dV)**2
                den = Ut_int**2 + Vt_int**2
    
                err = torch.sqrt(torch.sum(num) / (torch.sum(den)))
                RMS.append(err.item())
    return RMS
    
def RMS_per_window_new(Ut, Vt, Uo, Vo):
    U_true = Ut.detach().cpu().numpy()
    V_true = Vt.detach().cpu().numpy()

    U_other = Uo.detach().cpu().numpy()
    V_other = Vo.detach().cpu().numpy()
    
    # Merge the time dimensions: (days * steps_per_day, lon, lat)
    if U_other.ndim == 4:
        U_interp_flat = U_other.reshape(-1, *U_other.shape[2:])
        V_interp_flat = V_other.reshape(-1, *V_other.shape[2:])
        U_true_flat = U_true.reshape(-1, *U_true.shape[2:])
        V_true_flat = V_true.reshape(-1, *V_true.shape[2:])
    else:
        U_interp_flat = U_other
        V_interp_flat = V_other
        U_true_flat = U_true
        V_true_flat = V_true
    # Squared differences
    du_sq = (U_true_flat - U_interp_flat) ** 2
    dv_sq = (V_true_flat - V_interp_flat) ** 2
    num = du_sq + dv_sq

    # Squared norm of true velocity
    denom = U_true_flat**2 + V_true_flat**2

    # Sum over spatial dimensions (lon, lat)
    # num_sum = np.sum(num, axis=(1, 2))
    # denom_sum = np.sum(denom, axis=(1, 2))
    num_sum = np.nansum(num, axis=(1, 2))
    denom_sum = np.nansum(denom, axis=(1, 2))
    
    # Avoid division by zero
    denom_sum[denom_sum == 0] = np.nan

    # Time-dependent RMS error, output shape: (days * steps_per_day,)
    error = np.sqrt(num_sum / denom_sum)*100

    return error

def RMS_per_window(truth, other):
    # Loading truth field
    Truth_field = np.load(truth)
    U_true = Truth_field['U_int']
    V_true = Truth_field['V_int']
    
    # Loading other field
    Other_field = np.load(other)
    U_other = Other_field['U_int']
    V_other = Other_field['V_int']
    
    # Merge the time dimensions: (days * steps_per_day, lon, lat)
    U_interp_flat = U_other.reshape(-1, *U_other.shape[2:])
    V_interp_flat = V_other.reshape(-1, *V_other.shape[2:])
    U_true_flat = U_true.reshape(-1, *U_true.shape[2:])
    V_true_flat = V_true.reshape(-1, *V_true.shape[2:])

    # Squared differences
    du_sq = (U_true_flat - U_interp_flat) ** 2
    dv_sq = (V_true_flat - V_interp_flat) ** 2
    num = du_sq + dv_sq

    # Squared norm of true velocity
    denom = U_true_flat**2 + V_true_flat**2

    # Sum over spatial dimensions (lon, lat)
    num_sum = np.sum(num, axis=(1, 2))
    denom_sum = np.sum(denom, axis=(1, 2))

    # Avoid division by zero
    denom_sum[denom_sum == 0] = np.nan

    # Time-dependent RMS error, output shape: (days * steps_per_day,)
    error = np.sqrt(num_sum / denom_sum)*100
    return error
    

def multi_analysis(LON, LAT, obs_directory, field_directory, saving_directory, back_directory, lon0, lat0,
                   nDaysSim, lfil, dtAdvection, dtObservation, mylr, alpha, beta, 
                   n_drif, n_corr, toll, coefu=0, coefv=0, synth_traj=False, 
                   generate_background=False, device=None):
    torch.set_printoptions(precision=8)
    getcontext().prec = 8
    requires_grad = True
    dtype = torch.float32

    # assert nDaysSim/n_corr <=3, "Correction should be each 3 days."
    #Reading Truth
    if not field_directory.is_dir():
        raise FileNotFoundError(f"Truth directory not found: {field_directory}")

    filesList = sorted(field_directory.glob("*.nc"))

    Ndays = len(filesList)

    print(f'{Ndays} Truth files read')
    if nDaysSim > Ndays:
        nDaysSim = Ndays

    if nDaysSim <= 0:
        nDaysSim = Ndays

    # Read first field to have a sample
    Usample, Vsample, lon, lat = read_nc(filesList[0], LON, LAT, device=device, dtype=dtype)
    Nlon=lon.shape
    Nlat=lat.shape
    #get mask if needed
    maskuu, maskvv = get_mask(filesList[0], LON, LAT, device=device, dtype=dtype)

    fileu, filev, dx, dy, masku, maskv, coeffc, coeffd=GenerateCoefs(lat, lon, Nlat[0], Nlon[0], maskuu, maskvv, lfil, device=device, dtype=dtype)
    
    #Reading Background
    if not back_directory.is_dir():
        raise FileNotFoundError(f"Background directory not found: {back_directory}")

    filesListB = sorted(back_directory.glob("*.nc"))

    NdaysB = len(filesListB)

    print(f'{NdaysB} Background files read')
    if nDaysSim > NdaysB:
        nDaysSim = NdaysB

    if nDaysSim <= 0:
        nDaysSim = NdaysB

    # Read first field to have a sample
    UsampleB, VsampleB, lonB, latB = read_nc(filesListB[0], LON, LAT, device=device, dtype=dtype)
    
    uniform = False
    if nDaysSim%n_corr == 0:
        uniform = True
    days = nDaysSim//n_corr
    nObservationStepPerDay = int(24.0*3600.0/dtObservation)
    nAdvectionStepPerDay = int(24.0*3600.0/dtAdvection)
    
    RMS_back_t = []
    RMS_corr_t = []
    for window in range(n_corr):
        days_window = days
        if uniform == False and window == n_corr-1:
            days_window = nDaysSim%n_corr
        Ut_nograd = torch.zeros((days_window+1, *Usample.shape), device=device, dtype=dtype)
        Vt_nograd = torch.zeros((days_window+1, *Vsample.shape), device=device, dtype=dtype)

        Ut_nogradB = torch.zeros((days_window+1, *UsampleB.shape), device=device, dtype=dtype)
        Vt_nogradB = torch.zeros((days_window+1, *VsampleB.shape), device=device, dtype=dtype)
        
        for iFilename in range(days_window+1):
            Ut_nograd[iFilename], Vt_nograd[iFilename], _, _ = read_nc(filesList[iFilename+days*window], LON, LAT, 
                                                                       device=device, dtype=dtype, requires_grad=requires_grad)
            Ut_nogradB[iFilename], Vt_nogradB[iFilename], _, _ = read_nc(filesListB[iFilename+days*window], LON, LAT, 
                                                                         device=device, dtype=dtype, requires_grad=requires_grad)
            
        Ut = Ut_nograd.detach().requires_grad_()
        Vt = Vt_nograd.detach().requires_grad_()

        Ub = Ut_nogradB.detach().requires_grad_()
        Vb = Vt_nogradB.detach().requires_grad_()
            
        U_truth = Ut
        V_truth = Vt
        # Saving the truth
        truth_dir = saving_directory / f'truth{window+1}_{nDaysSim}days.npz'
        np.savez(truth_dir, Ut=Ut.detach().cpu().numpy(), Vt=Vt.detach().cpu().numpy()) 
        
        # Saving the background
        back_dir = saving_directory / f'background{window+1}_{nDaysSim}days.npz'
        np.savez(back_dir, Ut=Ub.detach().cpu().numpy(), Vt=Vb.detach().cpu().numpy()) 
        
        inter_truth = saving_directory / f'interpolated_truth{window+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
        field_int(U_truth, V_truth, dtAdvection, saving_file=inter_truth)
        
        
        if generate_background:
            # This is the U background
            Ut_nograd = coefu * Ut_nograd
            Vt_nograd = coefv * Vt_nograd

            Ub = Ut_nograd.detach().requires_grad_()
            Vb = Vt_nograd.detach().requires_grad_()
            
            # Saving the background
            back_dir = saving_directory / f'background{window+1}_{nDaysSim}days.npz'
            np.savez(back_dir, Ut=Ub.detach().cpu().numpy(), Vt=Vb.detach().cpu().numpy()) 
            
        inter_back = saving_directory / f'interpolated_back{window+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
        Ub_int, Vb_int = field_int(Ub, Vb, dtAdvection, saving_file=inter_back)
        RMS_back_t.extend(RMS_per_window(inter_truth, inter_back))
        

        lon_0 = torch.empty((n_drif), device=device, dtype=dtype) #stores initial lon for all drifters
        lat_0 = torch.empty((n_drif), device=device, dtype=dtype) #stores initial lat for all drifters

        obs_lon_trj = torch.empty((n_drif,days_window,nObservationStepPerDay), device=device, dtype=dtype) #stores lon of traj for all drifters
        obs_lat_trj = torch.empty((n_drif,days_window,nObservationStepPerDay), device=device, dtype=dtype) #stores lat of traj for all drifters

        #reading observations for all drifters
        for nd in range(1,n_drif+1):
            obs_path = obs_directory / f'obs0{nd}_{nDaysSim}days.npz'
            simobs = np.load(obs_path)
            lon_obs = torch.as_tensor(simobs["sim_lon"], device=device, dtype=dtype)
            lat_obs = torch.as_tensor(simobs["sim_lat"], device=device, dtype=dtype)

        
            obs_lon_trj[nd-1] = lon_obs[days*window:days*window+days_window]
            obs_lat_trj[nd-1] = lat_obs[days*window:days*window+days_window]
            
        
            lon_0[nd-1] = lon_obs[days*window][0]
            lat_0[nd-1] = lat_obs[days*window][0]
                
        if synth_traj and window==0:
            lon0_t = torch.as_tensor(lon0, device=device, dtype=dtype).flatten()
            lat0_t = torch.as_tensor(lat0, device=device, dtype=dtype).flatten()
            lon_0.copy_(lon0_t)
            lat_0.copy_(lat0_t)
        
        # The correction
        dU = torch.nn.Parameter(torch.zeros((1, *Usample.shape), device=device, dtype=dtype))
        dV = torch.nn.Parameter(torch.zeros((1, *Vsample.shape), device=device, dtype=dtype))
        
        optimizer = torch.optim.Adam([dU, dV], lr=mylr) 
        
        # Simulating drifters with the background #no correction
        for nd in range(1,n_drif+1):
            simfile = saving_directory / f'traj0{nd}_back{window+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
            lon_sim, lat_sim = simul_drifter(mylonIN = lon_0[nd-1].reshape(1),
                                             mylatIN = lat_0[nd-1].reshape(1), 
                                             Ut=Ub + dU, Vt=Vb + dV, lon=lon, lat=lat,
                                             dtAdvection=dtAdvection, simfile=simfile)
        
        count = 0
        diff_norm_perc = 100.0
        full_grad_norm = 4.0
        norm_grad_old = 2.0

        
        perc = []
        losstotal_values = []
        counts = []
        
        # Precompute this once (same logic as your int(step*(dtObservation//dtAdvection)))
        # assuming dtObservation and dtAdvection are integers in seconds and dtObservation % dtAdvection == 0
        stride = int(dtObservation // dtAdvection)
        sample_idx = torch.arange(0, nObservationStepPerDay, device=device) * stride  # shape (nObservationStepPerDay,)
        
        while (diff_norm_perc > toll):
            full_grad_old = full_grad_norm
        
            optimizer.zero_grad(set_to_none=True)
            
            dUFil, dVFil = ApplyFilter(dU, dV, Nlat[0], Nlon[0], fileu, filev, dx, dy, 
                                   masku, maskv, coeffc, coeffd, lfil)
            
            loss_drif = torch.zeros((), device=device, dtype=dtype)

            #simulating trajectories while correcting
            for nd in range(1,n_drif+1):
                simfile = saving_directory / f'disregrd{nd}_window{window+1}.npz'
                lon_sim, lat_sim = simul_drifter(mylonIN = lon_0[nd-1].reshape(1),
                                                 mylatIN = lat_0[nd-1].reshape(1), 
                                                 Ut=Ub + dUFil,
                                                 Vt=Vb + dVFil, 
                                                 lon=lon, lat=lat,
                                                 dtAdvection=dtAdvection, simfile=simfile)
                lon_adv = lon_sim
                lat_adv = lat_sim
                # Downsample WITHOUT Python loops (exact same selection)
                if dtAdvection != dtObservation:
                    lon_adv = lon_sim[:, sample_idx]
                    lat_adv = lat_sim[:, sample_idx]
                else:
                    lon_adv = lon_sim
                    lat_adv = lat_sim
                    
                y = torch.stack([lon_adv, lat_adv])
                yobs = torch.stack([obs_lon_trj[nd-1], obs_lat_trj[nd-1]])

                # convert lon/lat residuals (degrees) to meters locally
                # lat0_rad = (0.5*(lat_adv + yobs[1])).mean() * torch.pi/180.0 # ref lat
                # mx = R_earth * torch.cos(lat0_rad) * (torch.pi/180.0)  # meters per degree lon
                # my = R_earth * (torch.pi/180.0)                        # meters per degree lat

                # dx = (lon_adv - obs_lon_trj[nd-1]) * mx
                # dy = (lat_adv - obs_lat_trj[nd-1]) * my

                # loss = torch.mean(dx**2 + dy**2)   # meters^2 (already averaged)
                loss = torch.nn.functional.mse_loss(y, yobs)
                loss_drif = loss_drif + loss 
            # loss_drif = loss_drif / n_drif # Drifter misfit
            bg_pen = torch.norm(dU) + torch.norm(dV) # Background penalty
            div_pen = diverg_penalty(dU, dV, lat, lon) # Div penalty already normalized   
            #objective function
            losstotal = loss_drif + alpha * bg_pen + beta * div_pen
            
            # compute the gradient w.r.t to delta u
            losstotal.backward()

            norm_grad = float(torch.linalg.norm(dU.grad.detach()).item())
            
            # save the gradient dUt
            grad_path = saving_directory / f'grad_iterk_window_{window+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
            np.savez(grad_path, dUt=dU.grad.detach().cpu().numpy(), dVt=dV.grad.detach().cpu().numpy())
            
    
            # correct du, dv
            # this corresponds to du_new=du_old-lr*grad(du)
            optimizer.step()
            losstotal_values.append(float(losstotal.item()))
    
    
            # diff_norm_perc (do it in torch, then to float once)
            full_grad_norm = torch.sqrt(torch.linalg.norm(dU.grad)**2 + torch.linalg.norm(dV.grad)**2)
            diff_norm_perc = 100.0 * abs(full_grad_norm - full_grad_old) / (abs(full_grad_old)+1e-12)
            #diff_norm_perc = 100.0 * abs(norm_grad - norm_grad_old) / (abs(norm_grad_old) + 1e-12)
            print(f"diff_norm_perc = {diff_norm_perc}")

            perc.append(diff_norm_perc)

            count += 1
            print(f"iteration: {count} - window: {window+1}/{n_corr}")
            counts.append(count)
            
        print('while finished!')   
        if diff_norm_perc <= toll:
            # msd = 0
            dU_path = saving_directory / f'dUincrement_{window+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
            np.savez(dU_path, dUt=dU.detach().cpu().numpy(),
                     dVt=dV.detach().cpu().numpy())
            dU_int = torch.zeros((1, 1, *Usample.shape), device=device, dtype=dtype)
            dV_int = torch.zeros((1, 1, *Vsample.shape), device=device, dtype=dtype)
            dU_int[0] = dU
            dV_int[0] = dV

            corrected = saving_directory / f'corrected{window+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
            Uc_int = Ub_int + dU_int
            Vc_int = Vb_int + dV_int

            np.savez(corrected, U_int=Uc_int.detach().cpu().numpy(), V_int=Vc_int.detach().cpu().numpy())
            RMS_corr_t.extend(RMS_per_window(inter_truth, corrected))
            
            dUFil, dVFil = ApplyFilter(dU, dV, Nlat[0], Nlon[0], fileu, filev, dx, dy, 
                                   masku, maskv, coeffc, coeffd, lfil)
            dUFil_path = saving_directory / f'dUFilincrement_{window+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
            np.savez(dUFil_path, dUt=dUFil.detach().cpu().numpy(),
                     dVt=dVFil.detach().cpu().numpy())
            
            for nd in range(1,n_drif+1):
                simfile = saving_directory / f'traj_last{nd}_{window+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
                lon_sim, lat_sim = simul_drifter(mylonIN = lon_0[nd-1].reshape(1),
                                                 mylatIN = lat_0[nd-1].reshape(1), 
                                                 Ut=Ub + dUFil,
                                                 Vt=Vb + dVFil, 
                                                 lon=lon, lat=lat,
                                                 dtAdvection=dtAdvection, simfile=simfile)
                        
            print(f'done: time window {window+1}!!')
            obj_path = saving_directory / f'obj_fct_behavior_window{window+1}_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
            np.savez(obj_path, obj=np.array(losstotal_values), counts=np.array(counts))
          

    RMS_back_path = saving_directory / f'RMS_back_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'    
    np.savez(RMS_back_path, RMS_back_t=np.array(RMS_back_t))
    mean_back_rms = float(np.mean(RMS_back_t))

    RMS_corr_path = saving_directory / f'RMS_corr_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'    
    np.savez(RMS_corr_path, RMS_corr_t=np.array(RMS_corr_t))
    mean_corr_rms = float(np.mean(RMS_corr_t))
    drms_t = (np.array(RMS_back_t) - np.array(RMS_corr_t))/ np.array(RMS_back_t)
    mean_rdrms = np.mean(drms_t) * 100

    delta_rms = mean_back_rms - mean_corr_rms
    delta_rms_path = saving_directory / f'Delta_RMS_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
    np.savez(delta_rms_path, dRMS=delta_rms)

    #relative_dRMS = (mean_rdrms/mean_back_rms)*100.0
    relative_dRMS = mean_rdrms
    rdelta_rms_path = saving_directory / f'Relative_Delta_RMS_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
    np.savez(rdelta_rms_path, rdRMS=mean_rdrms)

    # Calculating MSD
    MSD = torch.zeros((), device=device, dtype=dtype)

    # precompute downsample indices once 
    stride = int(dtObservation // dtAdvection)
    sample_idx_obs = torch.arange(0, nObservationStepPerDay, device=device) * stride

    for nd in range(1, n_drif + 1):
        lon_traj = torch.empty((nDaysSim, nAdvectionStepPerDay), device=device, dtype=dtype)
        lat_traj = torch.empty((nDaysSim, nAdvectionStepPerDay), device=device, dtype=dtype)
    
        # stitch windows
        for window2 in range(n_corr):
            days_window = days
            if (uniform is False) and (window2 == n_corr - 1):
                days_window = nDaysSim % n_corr
    
            simfile = (saving_directory / (f"traj_last{nd}_{window2+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz"))
            simobs = np.load(simfile)
    
            lon_sim = torch.as_tensor(simobs["sim_lon"], device=device, dtype=dtype)
            lat_sim = torch.as_tensor(simobs["sim_lat"], device=device, dtype=dtype)
    
            lon_traj[days*window2 : days*window2 + days_window] = lon_sim
            lat_traj[days*window2 : days*window2 + days_window] = lat_sim
    
        # downsample if needed (no Python loops; same indices)
        if dtAdvection != dtObservation:
            lon_adv = lon_traj[:, sample_idx_obs]
            lat_adv = lat_traj[:, sample_idx_obs]
        else:
            lon_adv = lon_traj
            lat_adv = lat_traj
    
        # load obs for this drifter
        obs_path = obs_directory / f"obs0{nd}_{nDaysSim}days.npz"
        simobs = np.load(obs_path)
        lon_obs = torch.as_tensor(simobs["sim_lon"], device=device, dtype=dtype)
        lat_obs = torch.as_tensor(simobs["sim_lat"], device=device, dtype=dtype)
    
        y = torch.stack([lon_adv, lat_adv])          # (2, days, steps)
        yobs = torch.stack([lon_obs, lat_obs])       # same shape
        dist = torch.linalg.norm(y - yobs, dim=0)    # (days, steps)
    
        # Save dist(t) on CPU
        msd_t_path = saving_directory / (f"MSD(t)_drif{nd}_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz")
        np.savez(msd_t_path, dist_t=dist.detach().cpu().flatten().numpy())
    
        MSD = MSD + dist.mean()
    
    MSD = MSD / n_drif
    
    msd_path = saving_directory / (f"MSD_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz")
    np.savez(msd_path, mean_dist=float(MSD.item()))
            
    return MSD, mean_corr_rms, relative_dRMS 

def check_if_exists(name, grp): # for handling hdf5
    if name in grp:
        del grp[name]


def multi_analysis_optm(Ut, Vt, Ub, Vb, lon, lat, lon0, lat0, maskuu, maskvv, assim_grp, obs_grp, 
                   nDaysSim, lfil, dtAdvection, dtObservation, mylr, alpha, beta, 
                   n_drif, n_corr, toll, synth_traj=False,  device=None):
    torch.set_printoptions(precision=8)
    getcontext().prec = 8
    dtype = torch.float32

    dU_grp = assim_grp.require_group("dU")
    obj_funcion_grp = assim_grp.require_group("objective_function")
    err_grp = assim_grp.require_group("errors")

    Ut_all = Ut.detach().requires_grad_(True) # for grad computation ## in optimization loop
    Vt_all = Vt.detach().requires_grad_(True)

    Ub_all = Ub.detach().requires_grad_(True)
    Vb_all = Vb.detach().requires_grad_(True)

    Nlon=lon.shape
    Nlat=lat.shape
    fileu, filev, dx, dy, masku, maskv, coeffc, coeffd=GenerateCoefs(lat, lon, Nlat[0], Nlon[0], maskuu, maskvv, lfil, device=device, dtype=dtype)
    
    
    uniform = False
    if nDaysSim%n_corr == 0:
        uniform = True
    days = nDaysSim//n_corr
    nObservationStepPerDay = int(24.0*3600.0/dtObservation)
    nAdvectionStepPerDay = int(24.0*3600.0/dtAdvection)
    
    RMS_back_t = []
    RMS_corr_t = []
    for window in range(n_corr):
        days_window = days
        if uniform == False and window == n_corr-1:
            days_window = nDaysSim%n_corr
        
        Ut = Ut_all[window*days_window:window*days_window+days_window+1,:,:]
        Vt = Vt_all[window*days_window:window*days_window+days_window+1,:,:]

        Ub = Ub_all[window*days_window:window*days_window+days_window+1,:,:]
        Vb = Vb_all[window*days_window:window*days_window+days_window+1,:,:]

        
            
        Ut_int, Vt_int = field_int_new(Ut, Vt, dtAdvection)
        Ub_int, Vb_int = field_int_new(Ub, Vb, dtAdvection)
        
        
        RMS_back_t.extend(RMS_per_window_new(Ut_int, Vt_int, Ub_int, Vb_int))
        

        lon_0 = torch.empty((n_drif), device=device, dtype=dtype) #stores initial lon for all drifters
        lat_0 = torch.empty((n_drif), device=device, dtype=dtype) #stores initial lat for all drifters

        obs_lon_trj = torch.empty((n_drif,days_window,nObservationStepPerDay), device=device, dtype=dtype) #stores lon of traj for all drifters
        obs_lat_trj = torch.empty((n_drif,days_window,nObservationStepPerDay), device=device, dtype=dtype) #stores lat of traj for all drifters

        #reading observations for all drifters
        #read from .h5 file instead
        #open the group for observations and read lon/lat
        for nd in range(1,n_drif+1):

            lon_obs = obs_grp[f'lon_obs0{nd}_{nDaysSim}days'][:]
            lat_obs = obs_grp[f'lat_obs0{nd}_{nDaysSim}days'][:]

            lon_obs = torch.from_numpy(lon_obs).float().clone().to(device)
            lat_obs = torch.from_numpy(lat_obs).float().clone().to(device)

        
            obs_lon_trj[nd-1] = lon_obs[days*window:days*window+days_window]
            obs_lat_trj[nd-1] = lat_obs[days*window:days*window+days_window]
            
        
            lon_0[nd-1] = lon_obs[days*window][0]
            lat_0[nd-1] = lat_obs[days*window][0]
                
        if synth_traj and window==0:
            lon0_t = torch.as_tensor(lon0, device=device, dtype=dtype).flatten()
            lat0_t = torch.as_tensor(lat0, device=device, dtype=dtype).flatten()
            lon_0.copy_(lon0_t)
            lat_0.copy_(lat0_t)
        # The correction
        dU = torch.nn.Parameter(torch.zeros((1, *Ut[0].shape), device=device, dtype=dtype))
        dV = torch.nn.Parameter(torch.zeros((1, *Vt[0].shape), device=device, dtype=dtype))
        
        optimizer = torch.optim.Adam([dU, dV], lr=mylr) 
        traj_back_grp = assim_grp.require_group("traj_back")
        # Simulating drifters with the background #no correction
        for nd in range(1,n_drif+1):
            lon_back_name = f"lon_traj0{nd}_back{window+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}"
            lat_back_name = f"lat_traj0{nd}_back{window+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}"
            check_if_exists(lon_back_name, traj_back_grp)
            check_if_exists(lat_back_name, traj_back_grp)
            lon_back, lat_back = simul_drifter(mylonIN = lon_0[nd-1].reshape(1),
                                             mylatIN = lat_0[nd-1].reshape(1), 
                                             Ut=Ub + dU, Vt=Vb + dV, lon=lon, lat=lat,
                                             dtAdvection=dtAdvection)
            traj_back_grp.create_dataset(lon_back_name, data=lon_back.detach().cpu().numpy(), compression="gzip")
            traj_back_grp.create_dataset(lat_back_name, data=lat_back.detach().cpu().numpy(), compression="gzip")
           
        
        count = 0
        diff_norm_perc = 100.0
        full_grad_norm = 4.0
        full_grad_old = 2.0
        itr_max = 25

        
        perc = []
        losstotal_values = []
        counts = []
        
        # Precompute this once (same logic as int(step*(dtObservation//dtAdvection)))
        # assuming dtObservation and dtAdvection are integers in seconds and dtObservation % dtAdvection == 0
        stride = int(dtObservation // dtAdvection)
        sample_idx = torch.arange(0, nObservationStepPerDay, device=device) * stride  # shape (nObservationStepPerDay,)

        patience = 5          # number of iterations to wait
        eps_loss = 1e-4       # minimum meaningful improvement
        no_improve_count = 0
        flag = False

        best_loss = float("inf")
        
        while (diff_norm_perc > toll):
            full_grad_old = full_grad_norm
        
            optimizer.zero_grad(set_to_none=True)
            
            dUFil, dVFil = ApplyFilter(dU, dV, Nlat[0], Nlon[0], fileu, filev, dx, dy, 
                                   masku, maskv, coeffc, coeffd, lfil)
            
            loss_drif = torch.zeros((), device=device, dtype=dtype)

            #simulating trajectories while correcting
            for nd in range(1,n_drif+1):
                # simfile = saving_directory / f'disregrd{nd}_window{window+1}.npz'
                lon_sim, lat_sim = simul_drifter(mylonIN = lon_0[nd-1].reshape(1),
                                                 mylatIN = lat_0[nd-1].reshape(1), 
                                                 Ut=Ub + dUFil,
                                                 Vt=Vb + dVFil, 
                                                 lon=lon, lat=lat,
                                                 dtAdvection=dtAdvection)
                lon_adv = lon_sim
                lat_adv = lat_sim
                # Downsample WITHOUT Python loops (exact same selection)
                if dtAdvection != dtObservation:
                    lon_adv = lon_sim[:, sample_idx]
                    lat_adv = lat_sim[:, sample_idx]
                else:
                    lon_adv = lon_sim
                    lat_adv = lat_sim
                    
                y = torch.stack([lon_adv, lat_adv])
                yobs = torch.stack([obs_lon_trj[nd-1], obs_lat_trj[nd-1]])

                # convert lon/lat residuals (degrees) to meters locally
                # lat0_rad = (0.5*(lat_adv + yobs[1])).mean() * torch.pi/180.0 # ref lat
                # mx = R_earth * torch.cos(lat0_rad) * (torch.pi/180.0)  # meters per degree lon
                # my = R_earth * (torch.pi/180.0)                        # meters per degree lat

                # dx = (lon_adv - obs_lon_trj[nd-1]) * mx
                # dy = (lat_adv - obs_lat_trj[nd-1]) * my

                # loss = torch.mean(dx**2 + dy**2)   # meters^2 (already averaged)
                loss = torch.nn.functional.mse_loss(y, yobs)
                loss_drif = loss_drif + loss 
            # loss_drif = loss_drif / n_drif # Drifter misfit
            bg_pen = torch.norm(dU) + torch.norm(dV) # Background penalty
            div_pen = diverg_penalty(dU, dV, lat, lon) # Div penalty already normalized   
            #objective function
            losstotal = loss_drif + alpha * bg_pen + beta * div_pen
            
            # compute the gradient w.r.t to delta u
            losstotal.backward()
            
            # correct du, dv
            # this corresponds to du_new=du_old-lr*grad(du)
            optimizer.step()
            losstotal_values.append(float(losstotal.item()))

            current_loss = losstotal.item()

            if best_loss - current_loss > eps_loss:
                best_loss = current_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                
    
            # diff_norm_perc (do it in torch, then to float once)
            full_grad_norm = torch.sqrt(torch.linalg.norm(dU.grad)**2 + torch.linalg.norm(dV.grad)**2)
            diff_norm_perc = 100.0 * abs(full_grad_norm - full_grad_old) / (abs(full_grad_old)+1e-12)
            print(f"diff_norm_perc = {diff_norm_perc}")

            perc.append(diff_norm_perc)

            count += 1
            print(f"iteration: {count} - window: {window+1}/{n_corr}")

            if count > itr_max and no_improve_count >= patience:
                flag = True
                print("Early stopping: loss plateau")
                break
            counts.append(count)
            
        print('while finished!') 
        if diff_norm_perc <= toll or flag:
            # msd = 0
            #dU_path = saving_directory / f'dUincrement_{window+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
            du_name = f'dUincrement_{window+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}'
            dv_name = f'dVincrement_{window+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}'
            
            check_if_exists(du_name, dU_grp)
            check_if_exists(dv_name, dU_grp)
            dU_grp.create_dataset(du_name, data=dU.detach().cpu().numpy(), compression="gzip")
            dU_grp.create_dataset(dv_name, data=dV.detach().cpu().numpy(), compression="gzip")
            
            dU_int = torch.zeros((1, 1, *Ut[0].shape), device=device, dtype=dtype)
            dV_int = torch.zeros((1, 1, *Vt[0].shape), device=device, dtype=dtype)
            dU_int[0] = dU
            dV_int[0] = dV

            Uc_int = Ub_int + dU_int
            Vc_int = Vb_int + dV_int

            RMS_corr_t.extend(RMS_per_window_new(Ut_int, Vt_int, Uc_int, Vc_int))
            
            dUFil, dVFil = ApplyFilter(dU, dV, Nlat[0], Nlon[0], fileu, filev, dx, dy, 
                                   masku, maskv, coeffc, coeffd, lfil)
            
            
        traj_last_grp = assim_grp.require_group("traj_last")
        # Simulating drifters with the background #no correction
        for nd in range(1,n_drif+1):
            lon_last_name = f"lon_traj0{nd}_last{window+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}"
            lat_last_name = f"lat_traj0{nd}_last{window+1}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}"
            check_if_exists(lon_last_name, traj_last_grp)
            check_if_exists(lat_last_name, traj_last_grp)
            lon_last, lat_last = simul_drifter(mylonIN = lon_0[nd-1].reshape(1),
                                             mylatIN = lat_0[nd-1].reshape(1), 
                                             Ut=Ub + dUFil, Vt=Vb + dVFil, lon=lon, lat=lat,
                                             dtAdvection=dtAdvection)
            traj_last_grp.create_dataset(lon_last_name, data=lon_last.detach().cpu().numpy(), compression="gzip")
            traj_last_grp.create_dataset(lat_last_name, data=lat_last.detach().cpu().numpy(), compression="gzip")
                        
            print(f'done: time window {window+1}!!')
            obj_name = f'obj_fct_behavior_window{window+1}_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}'
            check_if_exists(obj_name, obj_funcion_grp)
            obj_funcion_grp.create_dataset(obj_name, data=np.array(losstotal_values), compression="gzip")
          

    
    RMS_back_name = f'RMS_back_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}'
    check_if_exists(RMS_back_name, err_grp)    
    err_grp.create_dataset(RMS_back_name, data=np.array(RMS_back_t), compression="gzip")


    RMS_corr_name = f'RMS_corr_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}'
    check_if_exists(RMS_corr_name, err_grp)      
    err_grp.create_dataset(RMS_corr_name, data=np.array(RMS_corr_t), compression="gzip")

    drms_t = (np.array(RMS_back_t) - np.array(RMS_corr_t))/ np.array(RMS_back_t)
    relative_dRMS = np.mean(drms_t) * 100

    rdelta_rms_name =f'Relative_Delta_RMS_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}'
    check_if_exists(rdelta_rms_name, err_grp)
    err_grp.create_dataset(rdelta_rms_name, data=relative_dRMS)
            
    return relative_dRMS 

def moving_window_corrections(LON, LAT, obs_directory, field_directory, back_directory, saving_directory, lon0, lat0,
                   nDaysSim, shift, lfil, dtAdvection, dtObservation, mylr, alpha, beta,
                   n_drif, Twindow, toll,  synth_traj,
                   generate_background, coefu=0, coefv=0, device=None):
    torch.set_printoptions(precision=8)
    getcontext().prec = 8
    requires_grad = True
    dtype = torch.float32

    # assert nDaysSim/n_corr <=3, "Correction should be each 3 days."
    #Reading Truth
    if not field_directory.is_dir():
        raise FileNotFoundError(f"Truth directory not found: {field_directory}")

    filesList = sorted(field_directory.glob("*.nc"))


    Ndays = len(filesList)

    print(f'{Ndays} files read')
    if nDaysSim > Ndays:
        nDaysSim = Ndays

    if nDaysSim <= 0:
        nDaysSim = Ndays

    # Read first field to have a sample
    Usample, Vsample, lon, lat = read_nc(filesList[0], LON, LAT, device=device, dtype=dtype)
    Nlon = lon.shape
    Nlat = lat.shape
    # get mask if needed
    maskuu, maskvv = get_mask(filesList[0], LON, LAT, device=device, dtype=dtype)

    fileu, filev, dx, dy, masku, maskv, coeffc, coeffd = GenerateCoefs(lat, lon, Nlat[0], Nlon[0], maskuu, maskvv, lfil, 
                                                                       device=device, dtype=dtype)

    nObservationStepPerDay = int(24.0 * 3600.0 / dtObservation)
    nAdvectionStepPerDay = int(24.0 * 3600.0 / dtAdvection)

    nWindows = (nDaysSim - Twindow) // shift + 1
    DU = torch.zeros((nWindows, 1, *Usample.shape), device=device, dtype=dtype)
    DV = torch.zeros((nWindows, 1, *Vsample.shape), device=device, dtype=dtype)

    k = 0
    start_pointer = k * shift
    end_pointer = start_pointer + Twindow
    final = nDaysSim

    while (end_pointer <= final):
        Ut_nograd = torch.zeros((Twindow + 1, *Usample.shape), device=device, dtype=dtype)
        Vt_nograd = torch.zeros((Twindow + 1, *Vsample.shape), device=device, dtype=dtype)

        for iFilename in range(Twindow + 1):
            Ut_nograd[iFilename], Vt_nograd[iFilename], _, _ = read_nc(filesList[iFilename + start_pointer], LON, LAT, 
                                                                       device=device, dtype=dtype,
                                                                       requires_grad=requires_grad)
        Ut = Ut_nograd.detach().requires_grad_()
        Vt = Vt_nograd.detach().requires_grad_()

        U_truth = Ut
        V_truth = Vt

        if generate_background:
            # This is the U background
            Ut_nograd = coefu * Ut_nograd
            Vt_nograd = coefv * Vt_nograd

            Ub = Ut_nograd.detach().requires_grad_()
            Vb = Vt_nograd.detach().requires_grad_()

            # Saving the background
            back_dir = saving_directory / f'background{k}_{nDaysSim}days.npz'
            np.savez(back_dir, Ut=Ub.detach().cpu().numpy(), Vt=Vb.detach().cpu().numpy())
        else:
            #Reading Background
            if not back_directory.is_dir():
                raise FileNotFoundError(f"Background directory not found: {back_directory}")

            filesListB = sorted(back_directory.glob("*.nc"))
        
            NdaysB = len(filesListB)
        
            print(f'{NdaysB} Background files read')
            if nDaysSim > Ndays:
                nDaysSim = Ndays
        
            if nDaysSim <= 0:
                nDaysSim = Ndays
        
            # Read first field to have a sample
            UsampleB, VsampleB, _, _ = read_nc(filesListB[0], LON, LAT)
            #get mask if needed
            maskuu, maskvv = get_mask(filesListB[0], LON, LAT)
            Ut_nogradB = torch.zeros((Twindow + 1, *UsampleB.shape), dtype=dtype)
            Vt_nogradB = torch.zeros((Twindow + 1, *VsampleB.shape), dtype=dtype)
            for iFilename in range(Twindow + 1):
                Ut_nogradB[iFilename], Vt_nogradB[iFilename], _, _ = read_nc(filesListB[iFilename + start_pointer], LON, LAT,
                                                                       requires_grad=requires_grad)

    
            Ub = Ut_nogradB.detach().requires_grad_()
            Vb = Vt_nogradB.detach().requires_grad_()
            # Saving the background
            back_dir = saving_directory / f'background{k}_{nDaysSim}days.npz'
            np.savez(back_dir, Ut=Ub.detach().cpu().numpy(), Vt=Vb.detach().cpu().numpy()) 

        lon_0 = torch.empty((n_drif), device=device, dtype=dtype)  # stores initial lon for all drifters
        lat_0 = torch.empty((n_drif), device=device, dtype=dtype)  # stores initial lat for all drifters

        obs_lon_trj = torch.empty((n_drif, Twindow, nObservationStepPerDay), device=device, dtype=dtype)  # stores lon of traj for all drifters
        obs_lat_trj = torch.empty((n_drif, Twindow, nObservationStepPerDay), device=device, dtype=dtype)  # stores lat of traj for all drifters

        # reading observations for all drifters
        for nd in range(1, n_drif + 1):
            obs_path = obs_directory / f'obs0{nd}_{nDaysSim}days.npz'
            simobs = np.load(obs_path)
            lon_obs = torch.as_tensor(simobs["sim_lon"], device=device, dtype=dtype)
            lat_obs = torch.as_tensor(simobs["sim_lat"], device=device, dtype=dtype)

            obs_lon_trj[nd - 1] = lon_obs[start_pointer:end_pointer]
            obs_lat_trj[nd - 1] = lat_obs[start_pointer:end_pointer]

            lon_0[nd - 1] = lon_obs[start_pointer][0]
            lat_0[nd - 1] = lat_obs[start_pointer][0]

        if synth_traj and start_pointer==0:
            lon0_t = torch.as_tensor(lon0, device=device, dtype=dtype).flatten()
            lat0_t = torch.as_tensor(lat0, device=device, dtype=dtype).flatten()
            lon_0.copy_(lon0_t)
            lat_0.copy_(lat0_t)

        # The correction
        dU = torch.nn.Parameter(torch.zeros((1, *Usample.shape), device=device, dtype=dtype))
        dV = torch.nn.Parameter(torch.zeros((1, *Vsample.shape), device=device, dtype=dtype))

        optimizer = torch.optim.Adam([dU, dV], lr=mylr)

        # Simulating drifters with the background #no correction
        for nd in range(1, n_drif + 1):
            simfile = saving_directory / f'traj0{nd}_back{k}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
            lon_sim, lat_sim = simul_drifter(mylonIN = lon_0[nd-1].reshape(1),
                                             mylatIN = lat_0[nd-1].reshape(1),
                                             Ut=Ub + dU, Vt=Vb + dV, lon=lon, lat=lat,
                                             dtAdvection=dtAdvection, simfile=simfile)

        count = 0
        diff_norm_perc = 100
        norm_grad = 4
        norm_grad_old = 2

        # Precompute this once (same logic as your int(step*(dtObservation//dtAdvection)))
        # assuming dtObservation and dtAdvection are integers in seconds and dtObservation % dtAdvection == 0
        stride = int(dtObservation // dtAdvection)
        sample_idx = torch.arange(0, nObservationStepPerDay, device=device) * stride  # shape (nObservationStepPerDay,)

        while (diff_norm_perc > toll):
            norm_grad_old = norm_grad

            optimizer.zero_grad()

            dUFil, dVFil = ApplyFilter(dU, dV, Nlat[0], Nlon[0], fileu, filev, dx, dy,
                                       masku, maskv, coeffc, coeffd, lfil)
            loss_drif = torch.tensor(0.0, device=dU.device)

            # simulating trajectories while correcting
            for nd in range(1, n_drif + 1):
                simfile = saving_directory / f'disregrd{nd}_window{k}.npz'
                lon_sim, lat_sim = simul_drifter(mylonIN = lon_0[nd-1].reshape(1),
                                                 mylatIN = lat_0[nd-1].reshape(1),
                                                 Ut=Ub + dUFil,
                                                 Vt=Vb + dVFil,
                                                 lon=lon, lat=lat,
                                                 dtAdvection=dtAdvection, simfile=simfile)
                lon_adv = lon_sim
                lat_adv = lat_sim
                # Downsample WITHOUT Python loops (exact same selection)
                if dtAdvection != dtObservation:
                    lon_adv = lon_sim[:, sample_idx]
                    lat_adv = lat_sim[:, sample_idx]
                else:
                    lon_adv = lon_sim
                    lat_adv = lat_sim

                y = torch.stack([lon_adv, lat_adv])
                yobs = torch.stack([obs_lon_trj[nd - 1], obs_lat_trj[nd - 1]])
                loss = torch.nn.functional.mse_loss(y, yobs)
                loss_drif = loss_drif + loss

            # objective function
            losstotal = (loss_drif + alpha * (torch.norm(dU) + torch.norm(dV))
                         + beta * diverg_penalty(dU, dV, lat, lon))

            # compute the gradient w.r.t to delta u
            losstotal.backward()

            # save the gradient dUt
            grad_path = saving_directory / f'grad_iterk_window_{k}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
            np.savez(grad_path, dUt=dU.grad.detach().cpu().numpy(), dVt=dV.grad.detach().cpu().numpy())

            gdUt = dU.grad.detach().numpy()
            norm_grad = LA.norm(gdUt)

            # correct du, dv
            # this corresponds to du_new=du_old-lr*grad(du)
            optimizer.step()
            #losstotal_values.append(losstotal.item())

            diff_norm_perc = 100 * (abs(norm_grad - norm_grad_old)) / (norm_grad_old)
            print(f'diff_norm_perc = {diff_norm_perc}')
            #perc.append(diff_norm_perc)
            count = count + 1
            print(f'iteration: {count} - window: {k + 1}/{nDaysSim - Twindow + 1}')
            #counts.append(count)

        print('while finished!')
        if diff_norm_perc <= toll:
            # msd = 0
            dU_path = saving_directory / f'dUincrement_{k}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
            np.savez(dU_path, dUt=dU.detach().cpu().numpy(),
                     dVt=dV.detach().cpu().numpy())

            dUFil, dVFil = ApplyFilter(dU, dV, Nlat[0], Nlon[0], fileu, filev, dx, dy,
                                       masku, maskv, coeffc, coeffd, lfil)
            dUFil_path = saving_directory / f'dUFilincrement_{k}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
            np.savez(dUFil_path, dUt=dUFil.detach().cpu().numpy(),
                     dVt=dVFil.detach().cpu().numpy())

            for nd in range(1, n_drif + 1):
                simfile = saving_directory / f'traj_last{nd}_{k}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
                lon_sim, lat_sim = simul_drifter(mylonIN = lon_0[nd-1].reshape(1),
                                                 mylatIN = lat_0[nd-1].reshape(1),
                                                 Ut=Ub + dUFil,
                                                 Vt=Vb + dVFil,
                                                 lon=lon, lat=lat,
                                                 dtAdvection=dtAdvection, simfile=simfile)

            print(f'done: time window {k + 1}!!')
        DU[k] = dU
        DV[k] = dV
        k = k + 1
        start_pointer = k * shift
        end_pointer = start_pointer + Twindow
    # MSD = 0
    # for nd in range(1, n_drif + 1):
    #     lon_traj = torch.empty((nDaysSim, nAdvectionStepPerDay))
    #     lat_traj = torch.empty((nDaysSim, nAdvectionStepPerDay))
    #     for window in range(nDaysSim-Twindow+1):
    #
    #         simfile = saving_directory + f'/traj_last{nd}_{k}window_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
    #         simobs = np.load(simfile)
    #         lon_sim, lat_sim = torch.tensor(simobs['sim_lon']), torch.tensor(simobs['sim_lat'])
    #
    #         lon_traj[start_pointer:end_pointer] = lon_sim
    #         lat_traj[start_pointer:end_pointer] = lat_sim
    #     lon_adv = lon_traj
    #     lat_adv = lat_traj
    #     if dtAdvection != dtObservation:
    #         store_lon = torch.empty((nDaysSim, nObservationStepPerDay))
    #         store_lat = torch.empty((nDaysSim, nObservationStepPerDay))
    #         for day in range(nDaysSim):
    #             for step in range(nObservationStepPerDay):
    #                 store_lon[day][step] = lon_traj[day][int(step * (dtObservation // dtAdvection))]
    #                 store_lat[day][step] = lat_traj[day][int(step * (dtObservation // dtAdvection))]
    #         lon_adv = store_lon
    #         lat_adv = store_lat
    #
    #     obs_path = obs_directory + f'/obs0{nd}_{nDaysSim}days.npz'
    #     simobs = np.load(obs_path)
    #     lon_obs, lat_obs = torch.tensor(simobs['sim_lon']), torch.tensor(simobs['sim_lat'])
    #
    #     y = torch.stack([lon_adv, lat_adv])
    #     yobs = torch.stack([lon_obs, lat_obs])
    #     delta_y = y - yobs
    #     dist = torch.norm(delta_y, dim=0)
    #
    #     # Saving dist in function of time
    #     msd_t_path = saving_directory + f'/MSD(t)_drif{nd}_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
    #     np.savez(msd_t_path, dist_t=dist.flatten().tolist())
    #
    #     mean_dist = torch.mean(dist)
    #
    #     MSD = MSD + mean_dist
    #
    # MSD = MSD / n_drif
    all_corr_path = saving_directory / f'dUArrays_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
    np.savez(all_corr_path, DU=DU.detach().cpu().numpy(), DV=DV.detach().cpu().numpy())

    return DU, DV

def moving_window_final(obs_directory, field_directory, back_directory, saving_directory, lon0, lat0,
                   nDaysSim, shift, lfil, dtAdvection, dtObservation, mylr, alpha, beta,
                   n_drif, Twindow, toll, coefu=0, coefv=0, synth_traj=False,
                   generate_background=False, plot_rms=True):
    torch.set_printoptions(precision=8)
    getcontext().prec = 8
    requires_grad = True
    dtype = torch.float64

    # assert nDaysSim/n_corr <=3, "Correction should be each 3 days."
    if not os.path.isdir(field_directory):
        field_directory = './data/aviso'

    filesList = glob.glob(os.path.join(field_directory, '*.nc'))
    filesList.sort()

    Ndays = len(filesList)

    print(f'{Ndays} aviso files read')
    if nDaysSim > Ndays:
        nDaysSim = Ndays

    if nDaysSim <= 0:
        nDaysSim = Ndays

    # Read first field to have a sample
    Usample, Vsample, lon, lat = read_nc(filesList[0], LON, LAT)
    Nlon = lon.shape
    Nlat = lat.shape

    Ut_nograd = torch.zeros((nDaysSim + 1, *Usample.shape), dtype=dtype)
    Vt_nograd = torch.zeros((nDaysSim + 1, *Vsample.shape), dtype=dtype)

    Uc_nograd = torch.zeros((nDaysSim + 1, *Usample.shape), dtype=dtype)
    Vc_nograd = torch.zeros((nDaysSim + 1, *Vsample.shape), dtype=dtype)

    for iFilename in range(nDaysSim + 1):
        Ut_nograd[iFilename], Vt_nograd[iFilename], _, _ = read_nc(filesList[iFilename],
                                                                   requires_grad=requires_grad)

    Ut = Ut_nograd.detach().requires_grad_()
    Vt = Vt_nograd.detach().requires_grad_()

    Uc = Uc_nograd.detach().requires_grad_()
    Vc = Vc_nograd.detach().requires_grad_()

    Uc = Uc.detach().numpy()
    Vc = Vc.detach().numpy()

    if generate_background:
        # This is the U background
        Ut_nograd = coefu * Ut_nograd
        Vt_nograd = coefv * Vt_nograd

        Ub0 = Ut_nograd.detach().requires_grad_()
        Vb0 = Vt_nograd.detach().requires_grad_()

        Ub = Ub0.detach().numpy()
        Vb = Vb0.detach().numpy()


    DU, DV = moving_window_corrections(obs_directory, field_directory, back_directory, saving_directory, lon0, lat0,
                   nDaysSim, shift, lfil, dtAdvection, dtObservation, mylr, alpha, beta,
                   n_drif, Twindow, toll,  synth_traj,
                   generate_background, coefu, coefv)
    DU = DU.detach().numpy()
    DV = DV.detach().numpy()
    t = 0
    Ucorr_term = np.zeros((1, *Usample.shape))
    Vcorr_term = np.zeros((1, *Vsample.shape))
    last = nDaysSim - Twindow # Last dU # Last index of array DU
    while t < Twindow:
        weights = [1 / (abs(t - k) + 1) for k in range(t,-1,-1)]
        norm = sum(weights)
        for j, k in enumerate(range(t,-1,-1)):
            corr_coef = weights[j] / norm
            Ucorr_term[0] += corr_coef * DU[k][0]
            Vcorr_term[0] += corr_coef * DV[k][0]
        Uc[t] = Ub[t] + Ucorr_term[0]
        Vc[t] = Vb[t] + Vcorr_term[0]
        t = t + 1 # AVISO gives data each 1 day
        Ucorr_term = np.zeros((1, *Usample.shape))
        Vcorr_term = np.zeros((1, *Vsample.shape))

    while (t >= Twindow and t <= last):
        weights = [1 / (abs(t - k) + 1) for k in range(t,t-Twindow,-1)]
        norm = sum(weights)
        for j, k in enumerate(range(t,t-Twindow,-1)):
            corr_coef = weights[j] / norm
            Ucorr_term[0] += corr_coef * DU[k][0]
            Vcorr_term[0] += corr_coef * DV[k][0]
        Uc[t] = Ub[t] + Ucorr_term[0]
        Vc[t] = Vb[t] + Vcorr_term[0]
        t = t + 1  # AVISO gives data each 1 day
        Ucorr_term = np.zeros((1, *Usample.shape))
        Vcorr_term = np.zeros((1, *Vsample.shape))

    while (t > last and t < nDaysSim):
        i = t - last
        weights = [1 / (abs(t - k) + 1) for k in range(last,Twindow+i,-1)]
        norm = sum(weights)
        for j, k in enumerate(range(last,Twindow+i,-1)):
            corr_coef = weights[j] / norm
            Ucorr_term[0] += corr_coef * DU[k][0]
            Vcorr_term[0] += corr_coef * DV[k][0]
        Uc[t] = Ub[t] + Ucorr_term[0]
        Vc[t] = Vb[t] + Vcorr_term[0]
        t = t + 1  # AVISO gives data each 1 day
        Ucorr_term = np.zeros((1, *Usample.shape))
        Vcorr_term = np.zeros((1, *Vsample.shape))
    Uc[nDaysSim] = Ub[nDaysSim] + DU[last][0]
    Vc[nDaysSim] = Vb[nDaysSim] + DV[last][0]

    Uc0 = torch.from_numpy(Uc).to(dtype=torch.float32).requires_grad_()
    Vc0 = torch.from_numpy(Vc).to(dtype=torch.float32).requires_grad_()

    corr_path = saving_directory + f'/UVcorr_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
    np.savez(corr_path, U=Uc, V=Vc)

    inter_truth = saving_directory + f'/(Moving)interpolated_truth_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
    field_int(Ut, Vt, dtAdvection, saving_file=inter_truth)

    inter_back = saving_directory + f'/(Moving)interpolated_back_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
    field_int(Ub0, Vb0, dtAdvection, saving_file=inter_back)

    inter_corr = saving_directory + f'/(Moving)interpolated_corr_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
    field_int(Uc0, Vc0, dtAdvection, saving_file=inter_corr)

    rms_back = RMS_per_window(truth=inter_truth, other=inter_back)
    rms_corr = RMS_per_window(truth=inter_truth, other=inter_corr)

    if plot_rms:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(0, len(rms_corr)), 100 * rms_corr, marker='o', color='black', linestyle='-',
                label='corrected')
        ax.legend()
        plt.title('RMS')
        plt.xlabel('Time (hours)')
        plt.ylabel('%RMS')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    return Uc, Vc

def moving_window_correct(Ub, Vb, DU, DV, Twindow, shift):
    """
    Apply moving-window reconstruction exactly as defined in the assimilation algorithm.

    Ub, Vb   : background velocity fields, shape (time, lon, lat)
    DU, DV   : list of window corrections (each constant over that window)
    Twindow  : duration of each window
    shift    : slide between windows (σ)
    """

    nDaysSim = Ub.shape[0]
    nWindows = DU.shape[0] 

    # Initialize corrected fields
    Uc = np.zeros_like(Ub)
    Vc = np.zeros_like(Vb)

    # Precompute window start, end, and centers
    window_start = np.array([k * shift for k in range(nWindows)])
    window_end   = window_start + Twindow
    window_center = window_start + Twindow / 2.0  # float center for weighting

    for t in range(nDaysSim):
        # Determine which windows overlap this time index
        valid_windows = [k for k in range(nWindows) if window_start[k] <= t < window_end[k]]

        if len(valid_windows) == 0:
            # No corrections apply → corrected = background
            Uc[t] = Ub[t]
            Vc[t] = Vb[t]
            print('no correction?')
            continue

        # Compute weights: inverse distance to window center
        distances = np.array([abs(t - window_center[k]) for k in valid_windows])
        weights = 1.0 / (distances + 1.0)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Weighted sum of corrections
        Ucorr = np.zeros_like(Ub[0])
        Vcorr = np.zeros_like(Vb[0])

        for w, k in zip(weights, valid_windows):
            Ucorr += w * DU[k][0]
            Vcorr += w * DV[k][0]

        # Apply correction
        Uc[t] = Ub[t] + Ucorr
        Vc[t] = Vb[t] + Vcorr

    return Uc, Vc

def moving_window_correct_eq5(Ub, Vb, DU, DV, Twindow, shift):
    """
    Eq.(5) reconstruction from Issa et al. (2016).

    Ub, Vb : (T, Ny, Nx) or (T, ...grid...)
    DU, DV : (nWindows, 1, Ny, Nx)  (constant correction per window)
    Twindow, shift : in same discrete time units as t index (e.g. "days")
    """
    T = Ub.shape[0]
    nWindows = DU.shape[0]

    # Window bounds in time-index units
    w_start = np.array([k * shift for k in range(nWindows)], dtype=int)
    w_end   = w_start + Twindow  # end is exclusive in code: t in [start, end)

    # window center in time index (can be half-integer)
    w_center = w_start + (Twindow / 2.0)

    Uc = np.empty_like(Ub)
    Vc = np.empty_like(Vb)

    for t in range(T):
        # windows that overlap time t
        valid = np.where((w_start <= t) & (t < w_end))[0]
        if valid.size == 0:
            Uc[t] = Ub[t]
            Vc[t] = Vb[t]
            continue

        # k* = window among valid whose center is closest to t
        k_star = valid[np.argmin(np.abs(w_center[valid] - t))]

        # Eq.(5) weights in WINDOW-INDEX space
        w = 1.0 / (np.abs(valid - k_star) + 1.0)
        w = w / w.sum()

        # weighted sum of constant corrections
        Ucorr = np.tensordot(w, DU[valid, 0], axes=(0, 0))
        Vcorr = np.tensordot(w, DV[valid, 0], axes=(0, 0))

        Uc[t] = Ub[t] + Ucorr
        Vc[t] = Vb[t] + Vcorr

    return Uc, Vc

def moving_window_final1(LON, LAT, obs_directory, field_directory, back_directory, saving_directory, lon0, lat0,
                   nDaysSim, shift, lfil, dtAdvection, dtObservation, mylr, alpha, beta,
                   n_drif, Twindow, toll, coefu=0, coefv=0, synth_traj=False,
                   generate_background=False, plot_rms=True, device=None):
    torch.set_printoptions(precision=8)
    getcontext().prec = 8
    requires_grad = True
    dtype = torch.float32

    # Velocity fields during complete experiment
    Ut0, Vt0, _, _ = read_VF(LON, LAT, field_directory, nDaysSim, lfil, device, dtype, requires_grad) #Truth

    if generate_background:
        Ub0, Vb0 = coefu * Ut0, coefv * Vt0
    else:
        Ub0, Vb0, _, _ = read_VF(LON, LAT, back_directory, nDaysSim, lfil, device, dtype, requires_grad)

    Ub, Vb = Ub0.detach().cpu().numpy(), Vb0.detach().cpu().numpy() #To numpy to apply smoothing #function requirement


    # Compute Windowed Corrections
    DU, DV = moving_window_corrections(LON, LAT, obs_directory, field_directory, back_directory, saving_directory, lon0, lat0,
                   nDaysSim, shift, lfil, dtAdvection, dtObservation, mylr, alpha, beta,
                   n_drif, Twindow, toll,  synth_traj,
                   generate_background, coefu, coefv,device)
    
    DU = DU.detach().cpu().numpy()
    DV = DV.detach().cpu().numpy()

    # Temporal Smoothing / Reconstruction
    Uc, Vc = moving_window_correct_eq5(Ub, Vb, DU, DV, Twindow, shift)

    # Save and Interpolate Results
    Uc0 = torch.from_numpy(Uc).to(device=device, dtype=dtype).requires_grad_() #Go back to torch for temporal interpolation
    Vc0 = torch.from_numpy(Vc).to(device=device, dtype=dtype).requires_grad_()

    corr_path = saving_directory / f'UVcorr_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
    np.savez(corr_path, U=Uc, V=Vc)

    inter_truth = saving_directory / f'(Moving)interpolated_truth_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
    field_int(Ut0, Vt0, dtAdvection, saving_file=inter_truth)

    inter_back = saving_directory / f'(Moving)interpolated_back_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
    field_int(Ub0, Vb0, dtAdvection, saving_file=inter_back)

    inter_corr = saving_directory / f'(Moving)interpolated_corr_{nDaysSim}days_fil{lfil}_alp{alpha:.6f}_bet{beta:.6f}_toll{toll:.2f}_lr{mylr:.6f}.npz'
    field_int(Uc0, Vc0, dtAdvection, saving_file=inter_corr)

    rms_back = RMS_per_window(truth=inter_truth, other=inter_back)
    rms_corr = RMS_per_window(truth=inter_truth, other=inter_corr)

    if plot_rms:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(0, len(rms_corr)), rms_corr, marker='o', color='black', linestyle='-',
                label='corrected')
        ax.plot(range(0, len(rms_corr)), rms_back, marker='o', color='blue', linestyle='-',
                label='background')
        ax.legend()
        plt.title('RMS')
        plt.xlabel('Time (hours)')
        plt.ylabel('%RMS')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return Uc, Vc

def read_VF(LON, LAT, field_directory, duration, lfil, device, dtype, requires_grad):
    if not field_directory.is_dir():
        raise FileNotFoundError(f"Field directory not found: {field_directory}")

    filesList = sorted(field_directory.glob("*.nc"))

    Ndays = len(filesList)

    print(f'{Ndays} Velocity fields - {field_directory} files read')
    if duration > Ndays:
        duration = Ndays

    if duration <= 0:
        duration = Ndays

    # Read first field to have a sample
    Usample, Vsample, lon, lat = read_nc(filesList[0], LON, LAT, device=device, dtype=dtype)
    Nlon=lon.shape
    Nlat=lat.shape
    #get mask if needed
    maskuu, maskvv = get_mask(filesList[0], LON, LAT, device=device, dtype=dtype)

    fileu, filev, dx, dy, masku, maskv, coeffc, coeffd=GenerateCoefs(lat, lon, Nlat[0], Nlon[0], maskuu, maskvv, lfil, device=device, dtype=dtype)
    Ut_nograd = torch.zeros((duration+1, *Usample.shape), device=device, dtype=dtype)
    Vt_nograd = torch.zeros((duration+1, *Vsample.shape), device=device, dtype=dtype)

    for iFilename in range(duration + 1):
        Ut_nograd[iFilename], Vt_nograd[iFilename], _, _ = read_nc(filesList[iFilename], LON, LAT, 
                                                                    device=device, dtype=dtype,
                                                                    requires_grad=requires_grad)
    Ut = Ut_nograd.detach().requires_grad_()
    Vt = Vt_nograd.detach().requires_grad_()

    return Ut, Vt, lon, lat

def run_onTruth(LON, LAT, nDaysSim, field_dir, saving_dir, dtAdvection, lon0, lat0, nDrifs, device=None):
    
    assert len(lon0) == len(lat0), "Size Mismatch - Check initial positions."
    assert len(lon0) == nDrifs, "Size Mismatch - Check number of drifters."

    requires_grad = False
    dtype = torch.float32

    
    directory = field_dir
    Ut, Vt, lon, lat = read_VF(LON, LAT, field_dir, nDaysSim, lfil=0, 
                    device=device, dtype=dtype, requires_grad=requires_grad)
    # VF diagnostics
    norm = torch.sqrt(torch.sum(Ut[0]**2 + Vt[0]**2))
    print(norm)
    mag = torch.sqrt(Ut[0]**2 + Vt[0]**2)  # shape (I, J)
    print("Mean speed (m/s):", mag.mean().item())
    print("Max speed (m/s):", mag.max().item())
    print("Min speed (m/s):", mag.min().item())
    

    for n in range(1, nDrifs + 1):
        mylon0 = torch.tensor([lon0[n-1]], device=device, dtype=dtype) #lon0 ,lat0 inputs are numpy lists
        mylat0 = torch.tensor([lat0[n-1]], device=device, dtype=dtype)
        simfile = saving_dir / f'obs0{n}_{nDaysSim}days.npz'
        simul_drifter (mylonIN=mylon0,
                       mylatIN=mylat0,
                        Ut=Ut, Vt=Vt, lon=lon, lat=lat, dtAdvection=dtAdvection, simfile=simfile)
    path = saving_dir / "positions.npz"
    np.savez(path, lat0=np.asarray(lat0), lon0=np.asarray(lon0))

    print("Trajectory simulated on the Truth!!")

def standardize_velocity(file_analysis, outfile=None):

    ds = xr.open_dataset(file_analysis)

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

    encoding = {"ugos": {"_FillValue": np.nan},"vgos": {"_FillValue": np.nan}}

    if outfile is not None:
        ds.to_netcdf(outfile, encoding=encoding)

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

def preprocessing(raw_truth_nc, raw_back_nc, out_truth, out_truth_intr, out_back):
    standardize_velocity(raw_back_nc, out_back)
    standardize_velocity(raw_truth_nc, out_truth)
    intrp_analysis_aviso1(out_truth, out_back, out_truth_intr)

    return 

def velocity_lon_lat(LON, LAT, start_time, end_time, truth, back):
    ds_truth = xr.open_dataset(truth, chunks={"time":end_time-start_time})
    ds_truth_sub = ds_truth.sel(longitude=slice(LON[0], LON[1]), latitude=slice(LAT[0], LAT[1]))
    dds_truth = ds_truth_sub.isel(time=slice(start_time,end_time))
    Ut = torch.from_numpy(dds_truth.ugos.values).float().transpose(1,2).clone()
    Vt = torch.from_numpy(dds_truth.vgos.values).float().transpose(1,2).clone()

    ds_back = xr.open_dataset(back, chunks={"time":end_time-start_time})
    ds_back_sub = ds_back.sel(longitude=slice(LON[0], LON[1]), latitude=slice(LAT[0], LAT[1]))
    dds_back = ds_back_sub.isel(time=slice(start_time,end_time))
    Ub = torch.from_numpy(dds_back.ugos.values).float().transpose(1,2).clone()
    Vb = torch.from_numpy(dds_back.vgos.values).float().transpose(1,2).clone()

    lon = torch.from_numpy(ds_back_sub.longitude.values).float()
    lat = torch.from_numpy(ds_back_sub.latitude.values).float()

    return Ub, Vb, Ut, Vt, lon, lat

def run_onTruth_new(Ut, Vt, lon, lat, nDaysSim, obs_grp, dtAdvection, lon0, lat0, nDrifs, device=None):
    
    assert len(lon0) == len(lat0), "Size Mismatch - Check initial positions."
    assert len(lon0) == nDrifs, "Size Mismatch - Check number of drifters."
    dtype = torch.float32

    # VF diagnostics
    norm = torch.sqrt(torch.sum(Ut[0]**2 + Vt[0]**2))
    print(norm)
    mag = torch.sqrt(Ut[0]**2 + Vt[0]**2)  # shape (I, J)
    print("Mean speed (m/s):", mag.mean().item())
    print("Max speed (m/s):", mag.max().item())
    print("Min speed (m/s):", mag.min().item())
    

    for n in range(1, nDrifs + 1):
        mylon0 = torch.tensor([lon0[n-1]], device=device, dtype=dtype) #lon0 ,lat0 inputs are numpy lists
        mylat0 = torch.tensor([lat0[n-1]], device=device, dtype=dtype)
        
        lon_name = f'lon_obs0{n}_{nDaysSim}days'
        lat_name = f'lat_obs0{n}_{nDaysSim}days'
        if lon_name in obs_grp:
            del obs_grp[lon_name]
        if lat_name in obs_grp:
            del obs_grp[lat_name]
        lon_sim, lat_sim = simul_drifter (mylonIN=mylon0, mylatIN=mylat0,
                        Ut=Ut, Vt=Vt, lon=lon, lat=lat, dtAdvection=dtAdvection)
        obs_grp.create_dataset(lon_name, data=lon_sim.detach().cpu().numpy(), compression="gzip")
        obs_grp.create_dataset(lat_name, data=lat_sim.detach().cpu().numpy(), compression="gzip")
        

    print("Trajectory simulated on the Truth!!")














    
    
    
    
    
    
    