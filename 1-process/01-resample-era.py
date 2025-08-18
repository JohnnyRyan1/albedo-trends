#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

1. Generate features (winter snowfall, summer air temperature)

2. Downscale ERA5 to ISMIP6 1 km grid.

"""


# Import modules
import xarray as xr
import numpy as np
from pyresample import kd_tree, geometry

#%%

# Define user
user = 'johnnyryan'

# Define path
path1 = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'
path2 = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/hydrology/data/'

# Import ISMIP 1 km grid
ismip_1km = xr.open_dataset(path2 + '1km-ISMIP6-GIMP.nc')

# Define target projection
target_def = geometry.SwathDefinition(lons=ismip_1km['lon'].values, lats=ismip_1km['lat'].values)

# Define ERA5 elevations
elev_file = xr.open_dataset(path1 + 'era5/era5-geopotential.nc')

# Convert geopotential height to elevation
elev = elev_file['z'] / 9.80665

# Read ERA5 data to find lat/lons
era_sf = xr.open_dataset(path1 + '/era5/data_stream-moda_stepType-avgad.nc')
era_t2m = xr.open_dataset(path1 + '/era5/data_stream-moda_stepType-avgua.nc')

# Extract lat/lon
era_lon, era_lat = np.meshgrid(era_sf['longitude'].values, era_sf['latitude'].values)

# Define source projection
source_def = geometry.SwathDefinition(lons=era_lon, lats=era_lat)


"""
Compute features

"""
# Select JJA months (June = 6, July = 7, August = 8)
jja = era_t2m.sel(valid_time=era_t2m.valid_time.dt.month.isin([6, 7, 8]))

# Group by year and compute summer mean
jja_t2m = jja['t2m'].groupby(jja.valid_time.dt.year).mean(dim='valid_time')
jja_t2m = jja_t2m.sel(year=jja_t2m.year != 1940)

# Select JJA months (June = 6, July = 7, August = 8)
jja = era_sf.sel(valid_time=era_sf.valid_time.dt.month.isin([6, 7, 8]))

# Group by year and compute summer mean
jja_sf = jja['sf'].groupby(jja.valid_time.dt.year).mean(dim='valid_time')
jja_sf = jja_sf.sel(year=jja_sf.year != 1940)

# Extract valid_time, month, year
t = era_sf.valid_time
month = t.dt.month
year = t.dt.year

# Define season year (Oct–May)
season_year = xr.DataArray(
    xr.where(month <= 5, year, year + 1),
    coords={'valid_time': t},
    dims='valid_time',
    name='season_year'
)

# Mask for Oct–May
oct_may_mask = (month >= 10) | (month <= 5)
sf_oct_may = era_sf['sf'].where(oct_may_mask)

# Count number of valid time steps per year
counts = sf_oct_may.groupby(season_year).count(dim='valid_time')  # this returns a DataArray

# Define expected time steps (monthly data: 8 months)
expected_steps = 8

# Create boolean mask for complete years (all grid points must have full 8-month record)
valid_years = counts >= expected_steps
valid_years = valid_years.all(dim=['latitude', 'longitude'])

# Clean season total: compute total snowfall and filter to valid years
snow_antecedent = sf_oct_may.groupby(season_year).mean(dim='valid_time')
snow_antecedent_clean = snow_antecedent.sel(season_year=valid_years.where(valid_years).dropna('season_year').season_year)

# Rename
snow_antecedent_clean = snow_antecedent_clean.rename({'season_year': 'year'})


"""
Downsample

"""

n_layers = snow_antecedent_clean.shape[0]

# Preallocate output array
resampled_sf = np.zeros((n_layers, target_def.shape[0], target_def.shape[1]))

# Loop and resample each 2D layer
for i in range(n_layers):
    print(i)
    resampled_sf[i] = kd_tree.resample_nearest(
        source_def,
        snow_antecedent_clean[i, :, :].values,
        target_def,
        radius_of_influence=50000,
        fill_value=np.nan
    )
    
# Preallocate output array
resampled_jja = np.zeros((n_layers, target_def.shape[0], target_def.shape[1]))

# Loop and resample each 2D layer
for i in range(n_layers):
    print(i)
    resampled_jja[i] = kd_tree.resample_nearest(
        source_def,
        jja_t2m[i, :, :].values,
        target_def,
        radius_of_influence=50000,
        fill_value=np.nan
    )

    
# Preallocate output array
resampled_jja_sf = np.zeros((n_layers, target_def.shape[0], target_def.shape[1]))

# Loop and resample each 2D layer
for i in range(n_layers):
    print(i)
    resampled_jja_sf[i] = kd_tree.resample_nearest(
        source_def,
        jja_sf[i, :, :].values,
        target_def,
        radius_of_influence=50000,
        fill_value=np.nan
    )


""" 
Save as NetCDF

"""

# Example: suppose resampled_sf.shape = (n_years, ny, nx)
n_years, ny, nx = resampled_sf.shape

# Build the coordinate arrays
years = snow_antecedent_clean.year.values
lons, lats = target_def.get_lonlats()

y_coords = np.mean(lats, axis=1)
x_coords = np.mean(lons, axis=0)

# Create DataArray
resampled_da = xr.DataArray(
    data=resampled_sf,
    dims=('year', 'y', 'x'),
    coords={'year': years, 'y': y_coords, 'x': x_coords},
    name='sf',
    attrs={'units': 'kg m-2'}
)

# Wrap in Dataset and save
resampled_ds = xr.Dataset({'sf': resampled_da})
resampled_ds.to_netcdf('/Users/johnnyryan/Desktop/era-winter-sf-1941-2025.nc')

# Create DataArray
resampled_da = xr.DataArray(
    data=resampled_jja,
    dims=('year', 'y', 'x'),
    coords={'year': years, 'y': y_coords, 'x': x_coords},
    name='t2m',
    attrs={'units': 'K'}
)

# Wrap in Dataset and save
resampled_ds = xr.Dataset({'t2m': resampled_da})
resampled_ds.to_netcdf('/Users/johnnyryan/Desktop/era-summer-t2m-1941-2025.nc')

# Create DataArray
resampled_da = xr.DataArray(
    data=resampled_jja_sf,
    dims=('year', 'y', 'x'),
    coords={'year': years, 'y': y_coords, 'x': x_coords},
    name='sf',
    attrs={'units': 'kg m-2'}
)

# Wrap in Dataset and save
resampled_ds = xr.Dataset({'sf': resampled_da})
resampled_ds.to_netcdf('/Users/johnnyryan/Desktop/era-summer-sf-1941-2025.nc')
#%%


