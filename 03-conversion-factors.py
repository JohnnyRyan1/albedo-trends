#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute correction factors for convert VNP and VJI to MCD.

"""

# Import packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Define user
user = 'jr555'

# Define filepath
path1 = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'
path2 = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/hydrology/data/'

# Define files
mcd = xr.open_dataset(path1 + 'satellite/mcd43a3-summer-albedo.nc')
vnp = xr.open_dataset(path1 + 'satellite/vnp43ma3-summer-albedo.nc')
vji = xr.open_dataset(path1 + 'satellite/vji43ma3-summer-albedo.nc')

# Air temp
t2m = xr.open_dataset(path1 + 'era5/era-summer-t2m-1941-2025.nc')

# Read mask
ismip_1km = xr.open_dataset(path2 + '1km-ISMIP6-GIMP.nc')
ice_mask = ismip_1km['GIMP'].values
mar_mask = xr.open_dataset(path1 + 'mar-masks.nc')
smb_mask = mar_mask['smb'].values
melt_mask = mar_mask['melt'].values

# Mask
mcd = mcd.where(ice_mask)
vnp = vnp.where(ice_mask)
vji = vji.where(ice_mask)

#%%

"""
Static conversion factor.

"""

# Compute mean difference between MCD and VNP
diff_vnp = mcd['albedo'][12:,:,:] - vnp['albedo']
diff_vji = mcd['albedo'][18:,:,:] - vji['albedo']

static_correction_vnp = np.mean(np.nanmean(diff_vnp, axis=(1,2)))
static_correction_vji = np.mean(np.nanmean(diff_vji, axis=(1,2)))

#%%

"""
Spatial conversion factor

"""

spatial_correction_vnp = np.nanmean(diff_vnp, axis=0)
spatial_correction_vji = np.nanmean(diff_vji, axis=0)

# Export as NetCDF
ds = xr.Dataset(
    {
        "conversion_vnp": (["y", "x"], spatial_correction_vnp),
        "conversion_vji": (["y", "x"], spatial_correction_vji)
    },
    coords={
        "y": ismip_1km['y'].values,
        "x": ismip_1km['x'].values
    },
    attrs={
        "description": "Spatial conversion factors for VNP43MA3 and VJI43MA3",
        "units": "unitless"
    }
)

# Save to NetCDF
ds.to_netcdf(path1 + "satellite/conversion-factors.nc")

#%%

temp = np.nanmean(t2m['t2m'][71:84,:,:], axis=(1,2))
d = np.nanmean(diff_vnp, axis=(1,2))

plt.scatter(np.arange(2012, 2025), d)
plt.scatter(temp, d)

#%%









