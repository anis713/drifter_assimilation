import math
from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
import sys
import os
import scipy
import torch
import numpy as np
import numpy.ma as ma
import glob
from tqdm import tqdm
from netCDF4 import Dataset as NetCDFFile

#from scipy.interpolate import RegularGridInterpolator
from multilinear import RegularGridInterpolator

R_earth = 6371229

def read_nc(filename, LON, LAT, device=None, dtype=torch.float32, requires_grad=False):
    with NetCDFFile(filename)as nc: #TODO: using xarray library
    # Assume the data are organize time,lat,lon
        lat = nc.variables['latitude'][:]
        masklat = (lat>=LAT[0]) & (lat<LAT[1])
        lon = nc.variables['longitude'][:]
        masklon = (lon>=LON[0]) & (lon<LON[1])
        lon = lon[masklon]
        lat = lat[masklat]
        U = nc.variables['ugos'][0,masklat,masklon].transpose()
        V = nc.variables['vgos'][0,masklat,masklon].transpose()
    U = torch.tensor(U.filled(0), dtype=dtype, device=device, requires_grad=requires_grad)
    V = torch.tensor(V.filled(0), dtype=dtype, device=device, requires_grad=requires_grad)
    lon = torch.tensor(lon, dtype=dtype, device=device)
    lat = torch.tensor(lat, dtype=dtype, device=device)
    
    return U, V, lon, lat
    
 
def get_mask(filename, LON, LAT, device=None, dtype=torch.float32, requires_grad=False):
    with NetCDFFile(filename)as nc: #TODO: using xarray library
    # Assume the data are organize time,lat,lon
        lat = nc.variables['latitude'][:]
        masklat = (lat>=LAT[0]) & (lat<LAT[1])
        lon = nc.variables['longitude'][:]
        masklon = (lon>=LON[0]) & (lon<LON[1])
        lon = lon[masklon]
        lat = lat[masklat]
        U = nc.variables['ugos'][0,masklat,masklon].transpose()
        V = nc.variables['vgos'][0,masklat,masklon].transpose()
    # maskuu=U.mask
    # maskvv=V.mask
    maskuu = ma.getmaskarray(U)
    maskvv = ma.getmaskarray(V)
    U = torch.tensor(U.filled(0), dtype=dtype, device=device, requires_grad=requires_grad)
    V = torch.tensor(V.filled(0), dtype=dtype, device=device, requires_grad=requires_grad)
    
    return maskuu, maskvv

def generate_drifters_positions(directory, lat_min, lat_max, lon_min, lon_max, N, erosion=1):
    if not os.path.isdir(directory):
        directory = './data/aviso'
    filesList = glob.glob(os.path.join(directory,'*.nc'))
    filesList.sort()
    
    # Read first field to have a sample
    Usample, Vsample, lon, lat = read_nc(filesList[0], LON, LAT, device)
        
    device = Usample.device
    
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing='xy')  
    
    # Step 1: Mask water points (non-NaN or non-zero velocities)
    water_mask = ~torch.isnan(Usample) & ~torch.isnan(Vsample)
    water_mask &= ((Usample.abs() + Vsample.abs()) > 0)
    
    # Step 2: Optional erosion to avoid coast (1-3 pixels usually enough)
    if erosion > 0:
        from scipy.ndimage import binary_erosion
        # Move to numpy for this step
        water_np = water_mask.cpu().numpy().astype(bool)
        eroded = binary_erosion(water_np, structure=torch.ones((3, 3)).numpy(), iterations=erosion)
        water_mask = torch.from_numpy(eroded).to(device)
    
    # Step 3: Flatten valid points
    valid_lats = lat_grid[water_mask]
    valid_lons = lon_grid[water_mask]
    
    # Step 5: Randomly sample N positions
    total_points = valid_lats.shape[0]
    print("Total valid water points:", total_points)
    if N > total_points:
        raise ValueError("Not enough valid water points to sample from.")

    indices = torch.randperm(total_points, device=device)[:N]
    initial_lat = valid_lats[indices]
    initial_lon = valid_lons[indices]
    
    return initial_lat, initial_lon


def generate_drifters_positions_anywhere(lat_min, lat_max, lon_min, lon_max, N, device='cpu'):
    """
    Randomly samples N initial positions uniformly over the lat-lon bounding box.
    """
    initial_lat = torch.empty(N, device=device).uniform_(lat_min, lat_max)
    initial_lon = torch.empty(N, device=device).uniform_(lon_min, lon_max)

    return initial_lat, initial_lon

def generate_nearby_drifters(lon, lat, num_drifters=100, delta=0.01):
    lon = np.array(lon).flatten()
    lat = np.array(lat).flatten()
    N = len(lon)

    drifter_lons = []
    drifter_lats = []

    for i in range(num_drifters):
        # Randomly pick an index along the trajectory (excluding endpoints for safety)
        i = np.random.randint(1, N-1)

        # Compute local tangent vector (finite difference)
        dx = lon[i+1] - lon[i-1]
        dy = lat[i+1] - lat[i-1]
        tangent = np.array([dx, dy])
        tangent /= np.linalg.norm(tangent)

        # Rotate 90 degrees to get normal vector
        normal = np.array([-tangent[1], tangent[0]])

        # Random displacement along the normal direction
        epsilon = np.random.uniform(-delta, delta)
        displacement = epsilon * normal

        # New drifter position
        new_lon = lon[i] + displacement[0]
        new_lat = lat[i] + displacement[1]

        drifter_lons.append(new_lon)
        drifter_lats.append(new_lat)

    return drifter_lons, drifter_lats



def generate_drifters(directory, lat_min, lat_max, lon_min, lon_max, N, 
                      nDaysSim, dtAdvection, erosion=1):
    requires_grad = False
    dtype = torch.float64
    if not os.path.isdir(directory):
        directory = './data/aviso'
        
    filesList = glob.glob(os.path.join(directory,'*.nc'))
    filesList.sort()

    Ndays=len(filesList)

    print(f'{Ndays} aviso files read')
    if nDaysSim>Ndays:
        nDaysSim = Ndays

    if nDaysSim<=0:
        nDaysSim = Ndays

    print(f'{nDaysSim} days simulated')


    # Read first field to have a sample
    Usample, Vsample, lon, lat = read_nc_new(filesList[0], lat_min, lat_max, lon_min, lon_max)

    Ut = torch.zeros((nDaysSim,*Usample.shape), dtype=dtype, requires_grad=requires_grad)
    Vt = torch.zeros((nDaysSim,*Vsample.shape), dtype=dtype,requires_grad=requires_grad)
    for iFilename in range(nDaysSim):
        Ut[iFilename], Vt[iFilename], _, _ = read_nc_new(filesList[iFilename], lat_min, lat_max, lon_min, lon_max,
                                                         requires_grad=requires_grad)
    
    initial_lat, initial_lon = generate_drifters_positions(directory, lat_min, lat_max, lon_min, lon_max, N)
    
    for drif in range(N):
        simfile = f'data_observed_{nDaysSim}Days/obs{drif+1}_{nDaysSim}Days.npz'
        simul_drifter(mylonIN=torch.tensor([initial_lon[drif].item()]), 
                      mylatIN=torch.tensor([initial_lat[drif].item()]), 
                      Ut=Ut, Vt=Vt, lon=lon, lat=lat, dtAdvection=dtAdvection, simfile=simfile)

def distance_two_trajectories(myfile1, myfile2, lon_arg, lat_arg):
    # Inputs: ------------------------------------------------------------------
    # myfile1:: type: String - file directory for first trajectory
    # myfile2:: type: String - file directory for second trajectory
    # lon_arg:: type: String - the argument for longitude in the files
    # lat_arg:: type: String - the argument for latitude in the files
    # Outputs: -----------------------------------------------------------------
    # err:: type: np.float -mean euclidien distance between the two trajectories 
    
    m_aviso = np.load(myfile1)
    m_back = np.load(myfile2)
    
    lon1, lat1 = m_aviso[lon_arg].reshape(-1), m_aviso[lat_arg].reshape(-1)
    lon2, lat2 = m_back[lon_arg].reshape(-1), m_back[lat_arg].reshape(-1)
    
    delta_lon = np.array(lon1) - np.array(lon2)
    delta_lat = np.array(lat1) - np.array(lat2)
    
    err = np.mean(np.sqrt(delta_lon**2+delta_lat**2))
    
    return err

def myinterpolate(U, lon, lat, mylon, mylat):
    interp = RegularGridInterpolator((lon,lat), U)
    pts=torch.stack([mylon,mylat])    
    return interp(pts)

#time interpolation
#t in seconds, a multiple of dt, is the time at which you want to interplorate between day1 and day2, in reference to day1
def timeinterpolate(U1, U2, V1, V2, t):
    U = U1 + (U2 - U1) * t /  (1 * 24 * 3600)
    V = V1 + (V2 - V1) * t / (1 * 24 * 3600)
    return U, V

def fromMetersToDeg(du,dv,lat_deg):
    lat_rad = lat_deg * torch.pi / 180.0
    dun = (180.0 / torch.pi) * du / (R_earth * torch.cos(lat_rad))
    dvn = (180.0 / torch.pi) * dv / R_earth
    return dun, dvn

def fromDegToMeters(mylon, mylat):
    phi = (2 * math.pi * mylat) / 360;
    theta = (2 * math.pi * mylon) / 360;
    x = R_earth * theta * math.cos(phi);
    y = R_earth * phi;
    return x, y


def interpolate2drifter(t, mylon, mylat, U1,U2,V1,V2, lon, lat):
    u1 = myinterpolate(U1, lon, lat, mylon,mylat)
    u2 = myinterpolate(U2, lon, lat, mylon,mylat)
    v1 = myinterpolate(V1, lon, lat, mylon,mylat)
    v2 = myinterpolate(V2, lon, lat, mylon,mylat)

    u1, v1 = timeinterpolate(u1, u2, v1, v2, t)
    #u1 = myinterpolate(Ut, lon, lat, mylon,mylat)
    #v1 = myinterpolate(Vt, lon, lat, mylon,mylat)
    return u1, v1

def avgVfield(Ut, Vt, nDaysSim):
    U = torch.transpose(Ut, 1, 2)
    V = torch.transpose(Vt, 1, 2)
    U_sum = torch.sum(U, dim=0)
    V_sum = torch.sum(V, dim=0)
    avgU = U_sum/nDaysSim
    avgV = V_sum/nDaysSim
    return avgU, avgV


def RK4(mylonIN, mylatIN, dtAdvection, t0, U11, U22, V11, V22, lon, lat):
    mylon = mylonIN
    mylat = mylatIN
    
    kwargs = dict(
        U1 = U11,
        U2 = U22,
        V1 = V11,
        V2 = V22,
        lon=lon,
        lat=lat)
    
    # step1
    u1, v1 = interpolate2drifter(t0, mylon, mylat, **kwargs)
    du1, dv1  = dtAdvection * u1, dtAdvection * v1
    dun1, dvn1 = fromMetersToDeg(du1, dv1, mylat)
    mylon, mylat = mylonIN + 0.5 * dun1, mylatIN + 0.5 * dvn1

    # Step 2
    u2, v2 = interpolate2drifter(t0 + 0.5 * dtAdvection, mylon, mylat, **kwargs)
    du2, dv2 = dtAdvection * u2, dtAdvection * v2
    dun2, dvn2 = fromMetersToDeg(du2, dv2, mylat)
    mylon, mylat = mylonIN + 0.5 * dun2, mylatIN + 0.5 * dvn2

    # Step 3
    u3, v3 = interpolate2drifter(t0 + 0.5 * dtAdvection, mylon, mylat, **kwargs)
    du3, dv3 = dtAdvection * u3, dtAdvection * v3
    dun3, dvn3 = fromMetersToDeg(du3, dv3, mylat)
    mylon, mylat = mylonIN + dun3, mylatIN + dvn3

    # Step 4
    u4, v4 = interpolate2drifter(t0 + dtAdvection, mylon, mylat, **kwargs)
    du4, dv4 = dtAdvection * u4, dtAdvection * v4
    dun4, dvn4 = fromMetersToDeg(du4, dv4, mylat)
    
    mylonOut = mylonIN + (dun1 + 2 * dun2 + 2 * dun3 + dun4) / 6
    mylatOut = mylatIN + (dvn1 + 2 * dvn2 + 2 * dvn3 + dvn4) / 6

    return mylonOut, mylatOut


def RK4_old(mylonIN, mylatIN, dtAdvection, t0, U11, U22, V11, V22, lon, lat):
    mylon = mylonIN
    mylat = mylatIN
    
    # step1
    Ut1, Vt1 = timeinterpolate(U11, U22, V11, V22, t0)

    u1 = myinterpolate(Ut1, lon, lat, mylon, mylat)
    v1 = myinterpolate(Vt1, lon, lat, mylon, mylat)
    du1 = dtAdvection * u1
    dv1 = dtAdvection * v1
    # normalize
    dun1, dvn1 = fromMetersToDeg(du1, dv1)
    mylon = mylonIN + 0.5 * dun1
    mylat = mylatIN + 0.5 * dvn1

    # Step 2
    Ut2, Vt2 = timeinterpolate(U11, U22, V11, V22, t0 + 0.5 * dtAdvection)
    u2 = myinterpolate(Ut2, lon, lat, mylon[0], mylat[0])
    v2 = myinterpolate(Vt2, lon, lat, mylon[0], mylat[0])
    du2 = dtAdvection * u2
    dv2 = dtAdvection * v2
    dun2, dvn2 = fromMetersToDeg(du2, dv2)
    mylon = mylonIN + 0.5 * dun2
    mylat = mylatIN + 0.5 * dvn2

    # Step 3
    Ut3 = Ut2
    Vt3 = Vt2
    u3 = myinterpolate(Ut3, lon, lat, mylon[0], mylat[0])
    v3 = myinterpolate(Vt3, lon, lat, mylon[0], mylat[0])
    du3 = dtAdvection * u3
    dv3 = dtAdvection * v3
    dun3, dvn3 = fromMetersToDeg(du3, dv3)
    mylon = mylonIN + dun3
    mylat = mylatIN + dvn3

    Ut4, Vt4 = timeinterpolate(U11, U22, V11, V22, t0 + dtAdvection)
    u4 = myinterpolate(Ut4, lon, lat, mylon[0], mylat[0])
    v4 = myinterpolate(Vt4, lon, lat, mylon[0], mylat[0])
    du4 = dtAdvection * u4
    dv4 = dtAdvection * v4
    dun4, dvn4 = fromMetersToDeg(du4, dv4)
    mylonOut = mylonIN + (dun1 + 2 * dun2 + 2 * dun3 + dun4) / 6
    mylatOut = mylatIN + (dvn1 + 2 * dvn2 + 2 * dvn3 + dvn4) / 6

    return mylonOut, mylatOut


def euler(mylonIN, mylatIN, dtAdvection, t0, U11, U22, V11, V22, lon, lat):
    mylon = mylonIN
    mylat = mylatIN
    # step1
    Ut1, Vt1 = timeinterpolate(U11, U22, V11, V22, t0)

    u1 = myinterpolate(Ut1, lon, lat, mylon, mylat)
    v1 = myinterpolate(Vt1, lon, lat, mylon, mylat)
    du1 = dtAdvection * u1
    dv1 = dtAdvection * v1
    # normalize
    dun1, dvn1 = fromMetersToDeg(du1, dv1)
    mylonOut = mylonIN + dun1
    mylatOut = mylatIN + dvn1

    return mylonOut, mylatOut

def diverg_penalty(dU, dV, lat, lon, eps=1e-12):
    """
    dU, dV: (Nlon, Nlat)  OR (1, Nlon, Nlat)  (your code uses dU[0], dV[0] anyway)
    lon: (Nlon,) degrees
    lat: (Nlat,) degrees
    returns: scalar torch tensor
    """
    # allow (1,Nlon,Nlat) input
    if dU.dim() == 3:
        dU = dU[0]
    if dV.dim() == 3:
        dV = dV[0]

    device = dU.device
    dtype  = dU.dtype

    lon = lon.to(device=device, dtype=dtype)
    lat = lat.to(device=device, dtype=dtype)

    # --- metric terms (cell sizes in meters) ---
    # dx depends on latitude (via cos(lat)), dy does not
    dlon = lon[1:] - lon[:-1]              # (Nlon-1,)
    dlat = lat[1:] - lat[:-1]              # (Nlat-1,)

    lat_rad = lat * torch.pi / 180.0       # (Nlat,)
    coslat  = torch.cos(lat_rad)           # (Nlat,)

    # dx(i,j) for i=0..Nlon-2, j=0..Nlat-1
    dx = R_earth * (2.0 * torch.pi / 360.0) * dlon[:, None] * coslat[None, :]   # (Nlon-1, Nlat)
    # dy(i,j) for i=0..Nlon-1, j=0..Nlat-2  (independent of i)
    dy = R_earth * (2.0 * torch.pi / 360.0) * dlat[None, :].expand(lon.numel(), -1)  # (Nlon, Nlat-1)

    # extend to full (Nlon,Nlat) by repeating last row/col 
    deltax = torch.empty((lon.numel(), lat.numel()), device=device, dtype=dtype)
    deltax[:-1, :] = dx
    deltax[-1,  :] = dx[-1, :]          # repeat last

    deltay = torch.empty((lon.numel(), lat.numel()), device=device, dtype=dtype)
    deltay[:, :-1] = dy
    deltay[:, -1 ] = dy[:, -1]          # repeat last

    # --- derivatives du/dx and dv/dy (central in interior, one-sided at edges) ---
    du_dx = torch.empty_like(dU)
    dv_dy = torch.empty_like(dV)

    # interior central differences
    du_dx[1:-1, :] = (dU[2:, :] - dU[:-2, :]) / (deltax[:-2, :] + deltax[2:, :] + eps)
    dv_dy[:, 1:-1] = (dV[:, 2:] - dV[:, :-2]) / (deltay[:, :-2] + deltay[:, 2:] + eps)

    # one-sided at boundaries 
    du_dx[0, :]  = (dU[1, :]  - dU[0, :])  / (deltax[0, :]  + eps)
    du_dx[-1, :] = (dU[-1, :] - dU[-2, :]) / (deltax[-1, :] + eps)

    dv_dy[:, 0]  = (dV[:, 1]  - dV[:, 0])  / (deltay[:, 0]  + eps)
    dv_dy[:, -1] = (dV[:, -1] - dV[:, -2]) / (deltay[:, -1] + eps)

    div_correction = du_dx + dv_dy
    return torch.linalg.norm(div_correction)   # scalar


def simul_drifter(mylonIN, mylatIN, Ut, Vt, lon, lat, dtAdvection, simfile=None):
    """Return the position of a drifter advected by a velocity field
    Parameters
    -----------
        mylonIN : torch tensor
            shape (1), contains the initial longitude of the drifter in degrees
        mylatIN : torch tensor
            shape (1), contains the initial latitude of the drifter in degrees
        Ut: torch tensor
            shape (Ndays, Nlon, Nlat), contains the u velocity field in m/s
        Vt: torch tensor
            shape (Ndays, Nlon, Nlat), contains the v velocity field in m/s
        lon: torch tensor
            shape (Nlon), contains the longitude values in degrees
        lon: torch tensor
            shape (Nlat), contains the latitudes values in degrees
        dtAdvection: float
            time between two advection steps in second
        simfile: str
            filename to save the output (default: None, nothing is saved)
    
    Outputs
    -------
        torch tensor
            shape (Ndays, nAdvectionStepPerDay), contains the successive longitude of the drifter in degrees
        torch tensor
            shape (Ndays, nAdvectionStepPerDay), contains the successive latitude of the drifter in degrees
            
    """
    device = Ut.device
                  
    nAdvectionStepPerDay = int(24.0*3600.0/dtAdvection)
    nDaysSim = Ut.shape[0]
    store_lon = torch.zeros((nDaysSim-1,nAdvectionStepPerDay), device=device)
    store_lat = torch.zeros((nDaysSim-1,nAdvectionStepPerDay), device=device)
    store_lon[0][0]=mylonIN
    store_lat[0][0]=mylatIN

    lon_min, lon_max = lon[0], lon[-1]
    lat_min, lat_max = lat[0], lat[-1]

    for iFilename in tqdm(range(0, nDaysSim-1)):
        if iFilename==0: #TODO: add a AVISO file for time=0
            Uday1, Vday1 = Ut[0], Vt[0]

        else:
            Uday1=Uday2
            Vday1=Vday2
        Uday2, Vday2 = Ut[iFilename+1], Vt[iFilename+1]

        for iAdvStepDay in range(0, nAdvectionStepPerDay):

            t0 = iAdvStepDay*dtAdvection

            mylonOut, mylatOut = RK4(mylonIN, mylatIN, dtAdvection, t0, U11=Uday1, U22=Uday2, V11=Vday1, V22=Vday2, lon=lon, lat=lat)

            cont_cond = ((mylonOut[0] > lon_max) | (mylonOut[0] < lon_min) |
                         (mylatOut[0] > lat_max) | (mylatOut[0] < lat_min))
            if cont_cond:
                print("A drifter is out of bound.\nOnly in-domain positions are saved.")
                break
            mylonIN = mylonOut
            mylatIN = mylatOut

            store_lon[iFilename][iAdvStepDay] = mylonIN[0]
            store_lat[iFilename][iAdvStepDay] = mylatIN[0]                                                                                                            
            #break

    if simfile:
        np.savez(simfile, sim_lon = store_lon.detach().cpu().numpy(), sim_lat = store_lat.detach().cpu().numpy())
    
    return store_lon, store_lat

def coefs_to_device(fileu, filev, dx, dy, masku, maskv, coeffc, coeffd, device, dtype=None):
    if dtype is None:
        dtype = fileu.dtype if torch.is_tensor(fileu) else torch.float32

    def _move(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device)  # for numpy arrays (masks)

    fileu = _move(fileu)
    filev = _move(filev)
    dx    = _move(dx)
    dy    = _move(dy)
    masku = _move(masku)
    maskv = _move(maskv)

    coeffc = {k: _move(v) for k, v in coeffc.items()}
    coeffd = {k: _move(v) for k, v in coeffd.items()}

    return fileu, filev, dx, dy, masku, maskv, coeffc, coeffd



def GenerateCoefs(lat, lon, Nlat, Nlon, maskuu, maskvv, lfil, device, dtype=None):
    deltax = torch.zeros((Nlon, Nlat))
    deltay = torch.zeros((Nlon, Nlat))
    fileu = torch.zeros((Nlon, Nlat))
    filev = torch.zeros((Nlon, Nlat))

    for i in range(0, Nlon - 1):
        for j in range(0, Nlat - 1):
            deltax[i][j] = R_earth * (2 * (math.pi) / 360) * (lon[i + 1] - lon[i]) * math.cos(
                2 * (math.pi) * lat[j] / 360)
            deltay[i][j] = R_earth * (2 * (math.pi) / 360) * (lat[j + 1] - lat[j])
    for j in range(0, Nlat):
        deltax[Nlon - 1][j] = deltax[Nlon - 2][j]
        deltay[Nlon - 1][j] = deltay[Nlon - 2][j]
    for i in range(0, Nlon):
        deltax[i][Nlat - 1] = deltax[i][Nlat - 2]
        deltay[i][Nlat - 1] = deltay[i][Nlat - 2]
    for i in range(0, Nlon):
        for j in range(0, Nlat):
            fileu[i][j] = 1. / math.sqrt(deltax[i][j] * deltay[i][j])
    filev = fileu

    masku = torch.zeros((Nlon, Nlat))
    maskv = torch.zeros((Nlon, Nlat))
    #maskv = torch.empty((Nlon, Nlat))

    mask = torch.zeros((Nlon, Nlat))
    for i in range(0, Nlon):
        for j in range(0, Nlat):
            mask[i][j] = (1 - int(maskuu[i][j])) * (1 - int(maskvv[i][j]))

    for i in range(0, Nlon - 1):
        for j in range(0, Nlat - 1):
            masku[i][j] = mask[i][j] * mask[i + 1][j]
            maskv[i][j] = mask[i][j] * mask[i][j + 1]



    #print(maskv)

    dx = deltax
    dy = deltay

    # should the below be torch ?? yes
    c1 = torch.zeros((Nlon, Nlat))
    c2 = torch.zeros((Nlon, Nlat))
    c3 = torch.zeros((Nlon, Nlat))
    c4 = torch.zeros((Nlon, Nlat))
    c5 = torch.zeros((Nlon, Nlat))
    c6 = torch.zeros((Nlon, Nlat))
    c7 = torch.zeros((Nlon, Nlat))
    c8 = torch.zeros((Nlon, Nlat))
    c9 = torch.zeros((Nlon, Nlat))

    d1 = torch.zeros((Nlon, Nlat))
    d2 = torch.zeros((Nlon, Nlat))
    d3 = torch.zeros((Nlon, Nlat))
    d4 = torch.zeros((Nlon, Nlat))
    d5 = torch.zeros((Nlon, Nlat))
    d6 = torch.zeros((Nlon, Nlat))
    d7 = torch.zeros((Nlon, Nlat))
    d8 = torch.zeros((Nlon, Nlat))
    d9 = torch.zeros((Nlon, Nlat))

    for i in range(1, Nlon - 1):
        for j in range(1, Nlat - 1):
            c1[i][j] = masku[i][j] * (
                        -1 / ((dx[i][j]) * (dx[i][j])) - dy[i][j] / (dx[i][j] * dx[i + 1][j] * dy[i + 1][j]) - mask[i][j - 1] * mask[i][
                        j] * mask[i + 1][j - 1] * mask[i + 1][j] * dx[i][j] / (dy[i][j] * dx[i][j - 1] * dy[i][j - 1]) -
                       mask[i][
                           j] * mask[i][j + 1] * mask[i + 1][j] * mask[i + 1][j + 1] / ((dy[i][j]) * (dy[i][j])))
            c2[i][j] = masku[i][j] * (1 / (dx[i][j] * dx[i + 1][j]))

            c3[i][j] = masku[i][j] * (
                        mask[i][j] * mask[i][j + 1] * mask[i + 1][j] * mask[i + 1][j + 1] * dx[i][j + 1] / (
                        ((dy[i][j]) * (dy[i][j])) * dx[i][j]))

            c4[i][j] = masku[i][j] * (
                        mask[i][j - 1] * mask[i][j] * mask[i + 1][j - 1] * mask[i + 1][j] / (dy[i][j] * dy[i][j - 1]))

            c5[i][j] = masku[i][j] * (dy[i - 1][j] / (dx[i][j] * dx[i][j] * dy[i][j]))

            c6[i][j] = masku[i][j] * (
                        -1 / (dx[i][j] * dy[i][j]) + mask[i][j] * mask[i][j + 1] * mask[i + 1][j] * mask[i + 1][
                    j + 1] / (
                                dx[i][j] * dy[i][j]))

            c7[i][j] = masku[i][j] * (
                        mask[i][j - 1] * mask[i][j] * mask[i + 1][j - 1] * mask[i + 1][j] * dy[i + 1][j - 1] / (
                        dy[i][j] * dx[i][j - 1] * dy[i][j - 1]) - dx[i + 1][j - 1] / (
                                dx[i][j] * dx[i + 1][j] * dy[i + 1][j]))  #

            c8[i][j] = masku[i][j] * (
                        -mask[i][j - 1] * mask[i][j] * mask[i + 1][j - 1] * mask[i + 1][j] / (dy[i][j] * dx[i][j - 1]) +
            dx[i][j - 1] / (dx[i][j] * dx[i][j] * dy[i][j]))

            c9[i][j] = masku[i][j] * (
                        -mask[i][j] * mask[i][j + 1] * mask[i + 1][j] * mask[i + 1][j + 1] * dy[i + 1][j] / (
                        dy[i][j] * dy[i][j] * dx[i][j]) + 1 / (dx[i][j] * dy[i + 1][j]))

            #########################################################################################################

            d1[i][j] = maskv[i][j] * (
        ((-mask[i - 1][j] * mask[i - 1][j + 1] * mask[i][j] * mask[i][j + 1] * dy[i][j]) / (
                dx[i][j] * dx[i - 1][j] * dy[i - 1][j]))
        - ((mask[i][j] * mask[i][j + 1] * mask[i + 1][j] *
            mask[i + 1][j + 1]) / (dx[i][j] * dx[i][j]))
        - (1 / (dy[i][j] * dy[i][j]))
        - (dx[i][j] / (
                dy[i][j] * dx[i][j + 1] * dy[i][j + 1]))
        )

            d2[i][j] = maskv[i][j] * (    mask[i][j] * mask[i][j + 1] * mask[i + 1][j] * mask[i + 1][j + 1] * dy[i + 1][j] /
                                  (dx[i][j] * dx[i][j] * dy[i][j])    )

            d3[i][j] = maskv[i][j] * (1 / (dy[i][j] * dy[i][j + 1]))

            d4[i][j] = maskv[i][j] * (dx[i][j - 1] / (dy[i][j] * dy[i][j] * dx[i][j]))
            d5[i][j] = maskv[i][j] * (
                    mask[i - 1][j] * mask[i - 1][j + 1] * mask[i][j] * mask[i][j + 1] / (dx[i][j] * dx[i - 1][j]))

            d6[i][j] = maskv[i][j] * (
                    -1 / (dx[i][j] * dy[i][j]) + mask[i][j] * mask[i][j + 1] * mask[i + 1][j] * mask[i + 1][j + 1] / (
                    dx[i][j] * dy[i][j]))

            d7[i][j] = maskv[i][j] * (-mask[i][j] * mask[i][j + 1] * mask[i + 1][j] * mask[i + 1][j + 1] * dx[i][j + 1] / (
                dx[i][j] * dx[i][j] * dy[i][j]) + 1 / (dy[i][j] * dx[i][j + 1]))

            d8[i][j] = maskv[i][j] * (
                    dy[i - 1][j] / (dy[i][j] * dy[i][j] * dx[i][j]) - mask[i - 1][j] * mask[i - 1][j + 1] * mask[i][j] *
    mask[i][j + 1] / (dx[i][j] * dy[i - 1][j]))
            d9[i][j] = maskv[i][j] * (
                -dy[i - 1][j + 1] / (dy[i][j] * dx[i][j + 1] * dy[i][j + 1]) + mask[i - 1][j] * mask[i - 1][j + 1] *
    mask[i][j] * mask[i][j + 1] * dx[i - 1][j + 1] / (dx[i][j] * dx[i - 1][j] * dy[i - 1][j]))


    # note ntfil/2=filter_order
    # rdtfil=dtfil
    # Uout=torch.empty((Nlon,Nlat))
    # Vout=torch.empty((Nlon,Nlat))

    coeffc = {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5, 'c6': c6, 'c7': c7, 'c8': c8, 'c9': c9}
    coeffd = {'d1': d1, 'd2': d2, 'd3': d3, 'd4': d4, 'd5': d5, 'd6': d6, 'd7': d7, 'd8': d8, 'd9': d9}

    fileu, filev, dx, dy, masku, maskv, coeffc, coeffd = coefs_to_device(
        fileu, filev, dx, dy, masku, maskv, coeffc, coeffd,
        device=device, dtype=dtype)


    return fileu, filev, dx, dy, masku, maskv, coeffc, coeffd

def ApplyFilter(dU, dV, Nlat, Nlon, fileu, filev, dx, dy, masku, maskv, coeffc, coeffd, lfil):
    """
    SAME algorithm (same update order + in-place). Assumes all inputs already on CUDA.
    """

    # quick safety checks (remove later if you want max speed)
    #assert dU.is_cuda and dV.is_cuda
    #assert fileu.is_cuda and masku.is_cuda
    #assert coeffc['c1'].is_cuda and coeffd['d1'].is_cuda

    # coeffs
    c1,c2,c3,c4,c5,c6,c7,c8,c9 = (coeffc[k] for k in ['c1','c2','c3','c4','c5','c6','c7','c8','c9'])
    d1,d2,d3,d4,d5,d6,d7,d8,d9 = (coeffd[k] for k in ['d1','d2','d3','d4','d5','d6','d7','d8','d9'])

    # filter order computed from dx/dy mins (one-time sync via .item() is OK)
    minx = float(dx.min().item())
    miny = float(dy.min().item())
    itxmin = math.floor((2.0 * lfil * lfil) / (minx * minx))
    itymin = math.floor((2.0 * lfil * lfil) / (miny * miny))
    filter_order = math.ceil((max(itxmin, itymin) + 1) / 2)

    nbitfil = filter_order * 2
    dtfil = 0.5 * lfil * lfil / nbitfil
    dtfil_t = torch.tensor(dtfil, device=dU.device, dtype=dU.dtype)

    U = torch.squeeze(dU * fileu)
    V = torch.squeeze(dV * filev)

    for _ in range(filter_order):
        for i in range(0, Nlon - 1):
            for j in range(0, Nlat - 1):
                U[i][j] = U[i][j] + dtfil_t * masku[i][j] * (
                    c1[i][j] * U[i][j] +
                    c2[i][j] * U[i + 1][j] +
                    c3[i][j] * U[i][j + 1] +
                    c4[i][j] * U[i][j - 1] +
                    c5[i][j] * U[i - 1][j] +
                    c6[i][j] * V[i][j] +
                    c7[i][j] * V[i + 1][j - 1] +
                    c8[i][j] * V[i][j - 1] +
                    c9[i][j] * V[i + 1][j]
                )

                V[i][j] = V[i][j] + dtfil_t * maskv[i][j] * (
                    d1[i][j] * V[i][j] +
                    d2[i][j] * V[i + 1][j] +
                    d3[i][j] * V[i - 1][j] +
                    d4[i][j] * V[i][j - 1] +
                    d5[i][j] * V[i][j + 1] +
                    d6[i][j] * U[i][j] +
                    d7[i][j] * U[i][j + 1] +
                    d8[i][j] * U[i - 1][j] +
                    d9[i][j] * U[i - 1][j + 1]
                )

    U = U / fileu
    V = V / filev

    return U.unsqueeze(0), V.unsqueeze(0)


def ApplyFilter_old(dU, dV, Nlat, Nlon, fileu, filev, dx, dy, masku, maskv, coeffc, coeffd, lfil,requires_grad=True):
    c1 = coeffc['c1']
    c2 = coeffc['c2']
    c3 = coeffc['c3']
    c4 = coeffc['c4']
    c5 = coeffc['c5']
    c6 = coeffc['c6']
    c7 = coeffc['c7']
    c8 = coeffc['c8']
    c9 = coeffc['c9']

    d1 = coeffd['d1']
    d2 = coeffd['d2']
    d3 = coeffd['d3']
    d4 = coeffd['d4']
    d5 = coeffd['d5']
    d6 = coeffd['d6']
    d7 = coeffd['d7']
    d8 = coeffd['d8']
    d9 = coeffd['d9']


    minx=int(torch.min(dx))
    miny=int(torch.min(dy))
    #minxy=int(torch.min(dx,dy))
    #itmin=math.floor((2 * lfil * lfil) / (minxy * minxy))
    #filter_order = int((itmin + 1) / 2)
    itxmin = math.floor((2 * lfil * lfil) / (minx * minx))
    itymin = math.floor((2 * lfil * lfil) / (miny * miny))
    filter_order = math.ceil((max(itxmin, itymin) + 1) / 2)
    # print(filter_order)
    nbitfil = filter_order * 2
    ##nb of filter passes
    dtfil = 0.5 * lfil * lfil / nbitfil
    ##filter time step

    # should also multiply by mask u, v respect.
    U = dU * fileu
    V = dV * filev

    U=torch.squeeze(U)
    V=torch.squeeze(V)

    for itfil in range(0, filter_order):
    # filhuv(uin*fileu, vin*filev)
        for i in range(0, Nlon - 1):
            for j in range(0, Nlat - 1):
                U[i][j] = U[i][j] + dtfil * masku[i][j] * (
                        c1[i][j] * U[i][j] + c2[i][j] * U[i + 1][j] + c3[i][j] * U[i][j + 1] + c4[i][j] * U[i][j - 1] +
                        c5[i][j] * U[i - 1][j] + c6[i][j] * V[i][j] + c7[i][j] * V[i + 1][j - 1] + c8[i][j] * V[i][
                            j - 1] + c9[i][j] * V[i + 1][j])
                V[i][j] = V[i][j] + dtfil * maskv[i][j] * (
                        d1[i][j] * V[i][j] + d2[i][j] * V[i + 1][j] + d3[i][j] * V[i - 1][j] + d4[i][j] * V[i][j - 1] +
                        d5[i][j] * V[i][j + 1] + d6[i][j] * U[i][j] + d7[i][j] * U[i][j + 1] + d8[i][j] * U[i - 1][j] + d9[i][
                            j] * U[i - 1][j + 1])

    U=U/fileu
    V=V/filev
    dUout=torch.unsqueeze(U,0)
    dVout =torch.unsqueeze(V,0)
    #print(dVout)
    return dUout, dVout
    #return torch.tensor(dUout, requires_grad=requires_grad), torch.tensor(dVout, requires_grad=requires_grad)
