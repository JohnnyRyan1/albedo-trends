#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Trends by station

"""

# Import packages
import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import linregress
from scipy.stats import ttest_rel
import glob
import os
import matplotlib.pyplot as plt

# Define user
user = 'johnnyryan'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'

# PROMICE
promice_files = sorted(glob.glob(path + 'promice/month/*.csv'))

# Satellite
mcd_points = xr.open_dataset(path + 'satellite/mcd43a3-points.nc')
vnp_points = xr.open_dataset(path + 'satellite/vnp43ma3-points.nc')
vji_points = xr.open_dataset(path + 'satellite/vji43ma3-points.nc')


#%%

station1, diff_before, diff_after, diff_overall = [], [], [], []
station2, aws_slope, aws_sig, mcd_slope, mcd_sig = [], [], [], [], []

for s in mcd_points['aws'].values:
    p = []
    for file in promice_files:
        filename = os.path.basename(file)
        if s + '_' in filename:
            p.append(file)

    site = pd.read_csv(p[0], index_col=['time'], parse_dates=['time'])
    site['aws'] = site['albedo']
    site_df = site['aws']
    site_mcd = mcd_points['albedo'].sel(aws=s).to_dataframe(name='mcd')
    site_mcd = site_mcd['mcd']
    site_df = pd.concat([site_df, site_mcd], axis=1)
    site_vnp = vnp_points['albedo'].sel(aws=s).to_dataframe(name='vnp')
    site_vnp = site_vnp['vnp']
    site_df = pd.concat([site_df, site_vnp], axis=1)
    site_vji = vji_points['albedo'].sel(aws=s).to_dataframe(name='vji')
    site_vji = site_vji['vji']
    site_df = pd.concat([site_df, site_vji], axis=1)
    site_df = site_df[site_df.index.month.isin([6, 7, 8])]
    site_summer = site_df.resample('Y').mean()
    site_count = site_df.resample('Y').count()
    site_summer[site_count != 3] = np.nan
    site_summer[site_summer['aws'] > 0.92] = np.nan
    
    # Compute differences
    diff = site_summer['aws'] - site_summer['mcd']
    diff_overall.append(np.nanmean(diff))
    diff_before.append(np.nanmean(diff['2000-01-01':'2020-01-01']))
    diff_after.append(np.nanmean(diff['2020-01-01':]))
    station1.append(s)
    
    # Compute trends
    x_aws = site_summer.index.year[(np.isfinite(site_summer['aws']) & (np.isfinite(site_summer['mcd'])))]
    y_aws = site_summer['aws'][(np.isfinite(site_summer['aws']) & (np.isfinite(site_summer['mcd'])))]
    x_mcd = site_summer.index.year[(np.isfinite(site_summer['aws']) & (np.isfinite(site_summer['mcd'])))]
    y_mcd = site_summer['mcd'][(np.isfinite(site_summer['aws']) & (np.isfinite(site_summer['mcd'])))]
    
    if (len(y_aws) & len(y_mcd)) >= 5:
        slope1, intercept1, r1, p1, se1 = linregress(x_aws, y_aws)
        aws_slope.append(slope1)
        aws_sig.append(p1)
        slope2, intercept2, r2, p2, se2 = linregress(x_mcd, y_mcd)
        mcd_slope.append(slope2)
        mcd_sig.append(p2)
        station2.append(s)
    else:
        station2.append(s)
        aws_slope.append(np.nan)
        aws_sig.append(np.nan)
        mcd_slope.append(np.nan)
        mcd_sig.append(np.nan)
    
mask = np.isfinite(np.array(diff_before)) & np.isfinite(np.array(diff_after))
a1, a2 = np.array(diff_before)[mask], np.array(diff_after)[mask]

t_stat, p_val = ttest_rel(a1, a2)
print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.3g}")

stats_df = pd.DataFrame(list(zip(station2, aws_slope, aws_sig, mcd_slope, mcd_sig)),
                        columns=['station', 'aws_slope', 'aws_sig', 'mcd_slope', 'mcd_sig'])

# Remove NaNs
stats_df = stats_df[np.isfinite(stats_df['aws_slope'])]

stats_df['agree'] = ((stats_df['aws_sig'] > 0.05) & (stats_df['mcd_sig'] > 0.05))
stats_df['direction'] = np.sign(stats_df['aws_slope']) == np.sign(stats_df['mcd_slope'])
(stats_df['agree'] == False) & (stats_df['direction'] == False)

(stats_df['agree'] == False) & (stats_df['direction'] == True)

#%%




#%%

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

# Set up subplots
fig, ax1 = plt.subplots(1, 1, figsize=(7, 4), layout='constrained')

ax1.plot(site_summer['aws'], color='black', lw=1.5, label='AWS', marker='o', linestyle='-', alpha=0.7)
ax1.plot(site_summer['mcd'], color=c1, lw=1.5, label='MCD43A3', marker='o', linestyle='-', alpha=0.7)
ax1.plot(site_summer['vnp'], color=c2, lw=1.5, label='VNP43MA3', marker='o', linestyle='-', alpha=0.7)
ax1.plot(site_summer['vji'], color=c4, lw=1.5, label='VJI43MA3', marker='o', linestyle='-', alpha=0.7)

ax1.legend(loc=3, fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True, which="both", linestyle="--", linewidth=1, zorder=1)
ax1.set_ylabel("Albedo", fontsize=12)  
ax1.set_xlim(pd.to_datetime('2000-06-01'),pd.to_datetime('2025-06-01'))

# Add station textbox
ax1.text(
    0.5, 0.98, s,
    transform=ax1.transAxes,
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='right'
)


#%%



x = site_summer.index.year[np.isfinite(site_summer['aws'])]
y = site_summer['aws'][np.isfinite(site_summer['aws'])]
slope, intercept, r, p, se = linregress(x, y)

# Format regression results
regression_text = (
    f"Slope: {slope:.3f}\n"
    f"p: {p:.3g}"
)

x = site_summer.index.year[np.isfinite(site_summer['mcd'])][19:]
y = site_summer['mcd'][np.isfinite(site_summer['mcd'])][19:]
slope, intercept, r, p, se = linregress(x, y)

# Format regression results
regression_text = (
    f"Slope: {slope:.3f}\n"
    f"p: {p:.3g}"
)









#%%