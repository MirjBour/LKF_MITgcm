#%% Utils Functions
from xmitgcm import open_mdsdataset
import sys
import numpy as np
import netCDF4 as nc
from scipy.interpolate import griddata
import xarray as xr
from dataset import *
from lkf_stats_tools import *
import pickle
#%% Logical Functions
def create_nc_file(a_ice_int, h_ice_int, u_ice_int, v_ice_int, shr_ice_int, div_ice_int, vor_ice_int, int_lons, int_lats, #int_dxu, int_dyv, 
                    ntimesteps, name):
    ds = nc.Dataset(name, 'w', format='NETCDF4')
    x = ds.createDimension('x', np.shape(a_ice_int)[2])
    y = ds.createDimension('y', np.shape(a_ice_int)[1])
    time = ds.createDimension('time', ntimesteps)
    x = ds.createVariable('x', 'f4', ('x',))
    y = ds.createVariable('y', 'f4', ('y',))
    time = ds.createVariable('time', 'f4', ('time'))
    a = ds.createVariable('A', 'f4', ('time','y','x'))
    h = ds.createVariable('H', 'f4', ('time','y','x'))
    u = ds.createVariable('U', 'f4', ('time','y','x'))
    v = ds.createVariable('V', 'f4', ('time','y','x'))
    div = ds.createVariable('div', 'f4', ('time','y','x'))
    shr = ds.createVariable('shr', 'f4', ('time','y','x'))
    vor = ds.createVariable('vor', 'f4', ('time','y','x'))
    lon = ds.createVariable('ULON', 'f4', ('y','x'))
    lat = ds.createVariable('ULAT', 'f4', ('y','x'))
    #dxu = ds.createVariable('DXU', 'f4', ('y','x'))
    #dyv = ds.createVariable('DYV', 'f4', ('y','x'))
    x[:] = np.arange(np.shape(a_ice_int)[2],dtype='int')
    y[:] = np.arange(np.shape(a_ice_int)[1],dtype='int')
    time[:] = np.arange(ntimesteps)
    a[:,:,:] = a_ice_int
    h[:,:,:] = h_ice_int
    u[:,:,:] = u_ice_int
    v[:,:,:] = v_ice_int
    shr[:,1:-1,1:-1] = shr_ice_int
    div[:,1:-1,1:-1] = div_ice_int
    vor[:,:-1,:-1] = vor_ice_int
    lon[:,:] = int_lons
    lat[:,:] = int_lats
    #dxu[:,:-1] = int_dxu
    #dyv[1:,:] = int_dyv
    ds.close()

def compute_strains(dxC,dyC,dyG,dxG, U, V, A):
    dxF = 0.5*(dxG[1:-1,1:-1]+dxG[1:-1,2:])
    dyF = 0.5*(dyG[1:-1,1:-1]+dyG[2:,1:-1])
    dyU = 0.5*(dyG[1:,:-1]+dyG[1:,1:])
    dxV = 0.5*(dxG[:-1,1:]+dxG[1:,1:])
    recip_dxF = 1/dxF
    recip_dyF = 1/dyF
    recip_dyU = 1/dyU
    recip_dxV = 1/dxV
    recip_dxF[np.isinf(recip_dxF)]=np.nan
    recip_dyF[np.isinf(recip_dyF)]=np.nan
    recip_dyU[np.isinf(recip_dyU)]=np.nan
    recip_dxV[np.isinf(recip_dxV)]=np.nan

    k1AtC = (dyG[2:,1:-1]-dyG[1:-1,1:-1])/(dxF*dyF)
    k2AtC = (dyG[1:-1,2:]-dyG[1:-1,1:-1])/(dxF*dyF)
    k1AtZ = (dyC[1:,1:]-dyC[:-1,1:])/(dyU*dxV)
    k2AtZ = (dxC[1:,1:]-dxC[1:,:-1])/(dxV*dyU)
    k1AtC[np.isinf(k1AtC)]=np.nan
    k2AtC[np.isinf(k2AtC)]=np.nan
    k1AtZ[np.isinf(k1AtZ)]=np.nan
    k2AtZ[np.isinf(k2AtZ)]=np.nan
    div = np.zeros((U.shape[0], U.shape[1]-2, U.shape[2]-2))
    shr = np.zeros((U.shape[0], U.shape[1]-2, U.shape[2]-2))
    vor = np.zeros((U.shape[0], U.shape[1]-1, U.shape[2]-1))

    for day in range(len(U)):
        # Compute velocity derivatives at C-points:
            dudx = (U[day,2:,1:-1]-U[day,1:-1,1:-1])*recip_dxF
            uave = 0.5*(U[day,2:,1:-1]+U[day,1:-1,1:-1])
            dvdy = (V[day,1:-1,2:]-V[day,1:-1,1:-1])*recip_dyF
            vave = 0.5*(V[day,1:-1,2:]+V[day,1:-1,1:-1])
            #print('velocity derivatives computed')
            
            # Compute strainrates at C-points:
            e11 = dudx + vave*k2AtC
            e22 = dvdy + uave*k1AtC
        
            # Compute velocity derivatives at Z-points:
            dudy = (U[day,1:,1:]-U[day,1:,:-1])*recip_dyU
            uave = 0.5*(U[day,1:,1:]+U[day,1:,:-1])
            dvdx = (V[day,1:,1:]-V[day,:-1,1:])*recip_dxV
            vave = 0.5*(V[day,1:,1:]+V[day,:-1,1:])
        
            # Compute strainrates at Z-points:
            e12z = 0.5*(dudy+dvdx)-k1AtZ*vave-k2AtZ*uave
        
            # Average four Z-points on one C-point:
            e12c = 0.25*(e12z[1:,1:]+e12z[1:,:-1]+e12z[:-1,1:]+e12z[:-1,:-1])
        
            # Compute strainrate invariants:
            div[day,:,:] = (e11+e22) * 24. * 3600. # units from s^-1 to day ^-1
            shr[day,:,:] = np.sqrt((e11-e22)**2+4*e12c**2) * 24. * 3600.
            vor[day,:,:] =  0.5*(dudy-dvdx) * 3600. *24.
    return div, vor, shr

def read_binary_file(file_path, shape, dtype='>f4'):
    """
    Reads a binary file and returns a numpy array.
    :param file_path: path to the binary file
    :param shape: shape of the output array
    :param dtype: data type of the binary file
    :return: numpy array with the specified shape and dtype
    """
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)
    return data.reshape(shape)

def get_lonlat_MITgcm(output_dir):
    lonc_file = output_dir + 'LONC.bin'
    latc_file = output_dir + 'LATC.bin'

    shape = (1536, 1680)
    # Reading binary files
    lonc = read_binary_file(lonc_file, shape)
    latc = read_binary_file(latc_file, shape)
    # Adding lonc and latc to the dataset
    ds['LONC'] = (('Y', 'X'), lonc)
    ds['LATC'] = (('Y', 'X'), latc)
    lon = ds.LONC.values
    lat = ds.LATC.values
    return(lon,lat)

def gen_lkf_dataset(root,simu,datatype,mask_rgps=False):
    if mask_rgps == True :
        nc_path = root + simu + '/MITgcm_4km_'+ simu[4:] + '.nc'
        output_path = root + simu +'/Dataset_Detected_LKFs'
        lkf = process_dataset(nc_path, output_path = output_path)
        lkf.detect_lkfs()
        lkf = lkf_dataset(True, output_path + '/MITgcm_4km_' + simu[4:] + '/', output_path + '/MITgcm_4km_' + simu[4:] + '/',
                                datatype, ['2008'], simu, read_tracking = False, mask_rgps = mask_rgps)
        lkf = process_dataset(nc_path,
                                    output_path = output_path)
        lkf.track_lkfs(force_recompute=True)
        lkf = lkf_dataset(True, output_path + '/MITgcm_4km_' + simu[4:] + '/', output_path + '/MITgcm_4km_' + simu[4:] + '/',
                                datatype, ['2008'], simu, read_tracking = True, mask_rgps = mask_rgps)
    else :
        nc_path = root + simu + '/MITgcm_4km_'+ simu[4:] + '.nc'
        output_path = root + simu +'/Dataset_Detected_LKFs'
        lkf = process_dataset(nc_path, output_path = output_path)
        lkf.track_lkfs(force_recompute=True)
        lkf = lkf_dataset(True, output_path + '/MITgcm_4km_' + simu[4:] + '/', output_path + '/MITgcm_4km_' + simu[4:] + '/',
                                datatype, ['2008'], simu, read_tracking = True, mask_rgps = mask_rgps)
    #lkf_data.gen_length()
    #lkf_data.gen_density()
    #lkf_data.gen_curvature()
    #lkf_data.gen_lifetime()
    lkf.gen_intersection()
    #lkf_data.gen_growthrate()
    with open(f"{output_path}/MITgcm_4km_{simu[4:]}/object_lkf_{simu[4:]}.pkl"  , 'wb') as f:
        pickle.dump(lkf, f)
    return lkf

#%% Script
root = '/scratch/users/evlema001/new_run/'
simus = ['run_elip', 'run_elip1', 'run_elip2', 'run_elip3', 'run_mohr', 'run_tear', 'run_tem']
lkf_simu={}
use_SIshear = False # True if you want to use the SIshear output files (recommended), False if you want to compute shear from U and V 

for simu in simus :
    output_dir = root + simu + '/' # path to the MITgcm output files
    ds = open_mdsdataset(output_dir)

    # Compute div, shr and vor
    (div,vor,shr) = compute_strains(ds.dxC.values,ds.dyC.values,ds.dyG.values,ds.dxG.values,ds.SIuice.values, ds.SIvice.values, ds.SIarea.values)
    if use_SIshear :
        shr = ds.SIshear.values
    lon,lat = get_lonlat_MITgcm(output_dir) # Get longitude and latitude from LONC.bin et LATC.bin
    nc_name = root + simu + '/MITgcm_4km_'+ simu[4:] +'.nc' # Set the name of the netCDF output file
    create_nc_file(ds.SIarea.values, ds.SIheff.values, ds.SIuice.values, ds.SIvice.values, shr, div, vor, lon, lat, #dxu, dyv, 
                len(ds.iter.values), nc_name )
    gen_lkf_dataset(root,simu,'mitgcm_4km', mask_rgps=True) # Generate LKF dataset from nc file and compute LKF statistics (see lkf_stats_tools.py)