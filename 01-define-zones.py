#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Produce mask that represents ablation and accumulation zones using MAR.

"""

# Import modules
import xarray as xr
import numpy as np
import glob
from pyresample import kd_tree, geometry

#%%

# Define user
user = 'jr555'

# Define path
path1 = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/hydrology/data/'
path2 = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'

# Import ISMIP 1 km grid
ismip_1km = xr.open_dataset(path1 + '1km-ISMIP6-GIMP.nc')

# Define mask
mask = ismip_1km['GIMP'].values

# Define MAR files
mar_files = sorted(glob.glob(path1 + 'mar/*.nc'))

smb_list = []
melt_list = []
for m in mar_files:
    # Read
    mar = xr.open_dataset(m)
    
    # Compute mean
    smb_list.append(mar['SMB'].mean(axis=0)[0,:,:].values)
    melt_list.append(mar['ME'].sum(axis=0)[0,:,:].values)

# Compute mean
mar_smb = np.mean(np.array(smb_list), axis=0)
mar_melt = np.sum((np.array(melt_list) == 0), axis=0)

#%%

"""
Regrid to ISMIP 1km

"""

# Define target projection
target_def = geometry.SwathDefinition(lons=ismip_1km['lon'].values, lats=ismip_1km['lat'].values)

# Define source projection
source_def = geometry.SwathDefinition(lons=mar['LON'].values, lats=mar['LAT'].values)

# Resample
resample_smb = kd_tree.resample_nearest(source_def, mar_smb, target_def, 
                                    radius_of_influence=50000, fill_value=np.nan)

# Resample
resample_melt = kd_tree.resample_nearest(source_def, mar_melt.astype('float'), target_def, 
                                    radius_of_influence=50000, fill_value=np.nan)

# Save as NetCDF
lons, lats = target_def.get_lonlats()

y_coords = np.mean(lats, axis=1)
x_coords = np.mean(lons, axis=0)

# Create DataArray
smb_da = xr.DataArray(
    data=resample_smb,
    dims=('y', 'x'),
    coords={'y': y_coords, 'x': x_coords},
    name='Mean SMB from MARv3.12',
    attrs={'units': 'mmWE/day'}
)    

# Create DataArray
melt_da = xr.DataArray(
    data=resample_melt,
    dims=('y', 'x'),
    coords={'y': y_coords, 'x': x_coords},
    name='Melt=0 years from MARv3.12',
    attrs={'units': 'mmWE/day'}
)                                    
     
# Wrap in Dataset and save
resampled_ds = xr.Dataset({'smb': smb_da, 'melt': melt_da})

#%%
resampled_ds.to_netcdf(path2 + 'mar-masks.nc')
                                 
                                      
                                      
                                      
                                      
                                      
                                      
                                      
                                      
                                      
                                      
                                      
                                      
  