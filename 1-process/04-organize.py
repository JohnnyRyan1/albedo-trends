#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Export AWS albedo data.

"""

# Import packages
import numpy as np
import pandas as pd
import xarray as xr
import glob

# Define user
user = 'jr555'

# Define filepath
path1 = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/hydrology/data/'
path2 = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'

# Define albedo data
mcd = xr.open_dataset(path2 + 'satellite/mcd43a3-summer-albedo.nc')
vnp = xr.open_dataset(path2 + 'satellite/vnp43ma3-summer-albedo.nc')
vji = xr.open_dataset(path2 + 'satellite/vji43ma3-summer-albedo.nc')

# Read mask
ismip_1km = xr.open_dataset(path1 + '1km-ISMIP6-GIMP.nc')
mask = ismip_1km['GIMP'].values

# AWS locations
aws_meta = pd.read_csv(path2 +'promice/AWS_sites_metadata.csv')
aws_meta = aws_meta[aws_meta['location_type'] == 'ice sheet']
point_ids = aws_meta['site_id'].values

# PROMICE
promice_files = sorted(glob.glob(path2 + 'promice/month/*.csv'))

# Generate list of ice sheet AWS
promice_ice_sheet = []
for i in range(len(aws_meta)):
    for file in promice_files:
        if aws_meta.iloc[i]['site_id'] + '_' in file:
            promice_ice_sheet.append(file)

# Define air temperature data
era = xr.open_dataset(path2 + 'era5/era-summer-t2m-1941-2025.nc')
era_winter_sf = xr.open_dataset(path2 + 'era5/era-summer-sf-1941-2025.nc')
era_summer_sf = xr.open_dataset(path2 + 'era5/era-winter-sf-1941-2025.nc')

lons, lats = [], []
for a in range(len(aws_meta)):
    # Find lat and lon
    lat, lon = aws_meta.iloc[a]['latitude_last_valid'], aws_meta.iloc[a]['longitude_last_valid']
    
    dist_sq = (ismip_1km['lat'].values - lat)**2 + (ismip_1km['lon'].values - lon)**2
    i, j = np.unravel_index(np.argmin(dist_sq), ismip_1km['lat'].values.shape)
    lons.append(j)
    lats.append(i)

#%%

# Sample values at given lat/lon indices
mcd_point_values = [mcd['albedo'][:, lat, lon].values for lat, lon in zip(lats, lons)]
vnp_point_values = [vnp['albedo'][:, lat, lon].values for lat, lon in zip(lats, lons)]
vji_point_values = [vji['albedo'][:, lat, lon].values for lat, lon in zip(lats, lons)]

modis_dates = pd.to_datetime([f"{year}-12-31" for year in mcd['albedo']['time'].values])
vnp_dates = pd.to_datetime([f"{year}-12-31" for year in vnp['albedo']['time'].values])
vji_dates = pd.to_datetime([f"{year}-12-31" for year in vji['albedo']['time'].values])
era_dates = pd.to_datetime([f"{year}-12-31" for year in era['year'][59:84].values])

t2m_point_values = [era['t2m'][59:84, lat, lon].values for lat, lon in zip(lats, lons)]
sf_w_point_values = [era_winter_sf['sf'][59:84, lat, lon].values for lat, lon in zip(lats, lons)]
sf_s_point_values = [era_summer_sf['sf'][59:84, lat, lon].values for lat, lon in zip(lats, lons)]

aws = []
for p in promice_ice_sheet:
    site = pd.read_csv(p, index_col=['time'], parse_dates=['time'])
    site['aws'] = site['albedo']
    site_df = site['aws']
    site_df = site_df[site_df.index.month.isin([6, 7, 8])]
    site_summer = site_df.resample('YE').mean()
    site_count = site_df.resample('YE').count()
    site_summer[site_count != 3] = np.nan
    site_summer[site_summer > 0.92] = np.nan
    aws.append(site_summer)
    
# Export as csv
for i in range(len(aws_meta)):
    # Get station name
    name = aws_meta.iloc[i]['site_id']
    
    # Define DataFrame
    df = pd.DataFrame(data=t2m_point_values[i],index=era_dates, columns=["t2m"])

    # Append other products
    df['aws'] = pd.DataFrame(aws[i])
    df['mcd'] = pd.DataFrame(data=mcd_point_values[i],index=modis_dates,  columns=["mcd"])
    df['vnp'] = pd.DataFrame(data=vnp_point_values[i],index=vnp_dates,  columns=["vnp"])
    df['vji'] = pd.DataFrame(data=vji_point_values[i],index=vji_dates,  columns=["vji"])

    df['sf_winter'] = pd.DataFrame(data=sf_w_point_values[i],index=era_dates, columns=["era"])
    df['sf_summer'] = pd.DataFrame(data=sf_s_point_values[i],index=era_dates, columns=["era"])
    
    # Set index name
    df.index.name = 'datetime'
    
    # Export
    df.to_csv(path2 + 'station/' + name + '.csv')


#%%

t2m_point_values = [era['t2m'][:, lat, lon].values for lat, lon in zip(lats, lons)]
sf_w_point_values = [era_winter_sf['sf'][:, lat, lon].values for lat, lon in zip(lats, lons)]
sf_s_point_values = [era_summer_sf['sf'][:, lat, lon].values for lat, lon in zip(lats, lons)]
era_dates = pd.to_datetime([f"{year}-12-31" for year in era['year'].values])

# Export as csv
for i in range(len(aws_meta)):
    # Get station name
    name = aws_meta.iloc[i]['site_id']
    
    # Define DataFrame
    df = pd.DataFrame(data=t2m_point_values[i],index=era_dates, columns=["t2m"])
    df['sf_winter'] = pd.DataFrame(data=sf_w_point_values[i],index=era_dates, columns=["era"])
    df['sf_summer'] = pd.DataFrame(data=sf_s_point_values[i],index=era_dates, columns=["era"])
    
    # Set index name
    df.index.name = 'datetime'
    
    # Export
    df.to_csv(path2 + 'era5/station/' + name + '.csv')












