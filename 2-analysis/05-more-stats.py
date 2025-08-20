#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

More stats

"""

# Import packages
import pandas as pd
import numpy as np
from scipy.stats import linregress
import glob
import os
import pymannkendall as mk
from scipy.stats import theilslopes
import matplotlib.pyplot as plt

# Define user
user = 'johnnyryan'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/'

# Define files
hindcast_files = sorted(glob.glob(path + 'data/hindcast/*.csv'))
modern_files = sorted(glob.glob(path + 'data/station/*.csv'))
mcd_trends = pd.read_csv(path + 'data/trends.csv')

# Sort by slope
df_sorted = mcd_trends.sort_values(by="elevation").reset_index(drop=True)

# Define order
order = df_sorted['station'].tolist()

# Read GBI
gbi = pd.read_csv(path + 'data/indices/gbi-summer.csv', index_col=['Date'], parse_dates=['Date'])
gbi_rolling = gbi.rolling(window=10, center=True, min_periods=1).mean()

#%%

"""
Compute slopes in albedo for each 10-year window

"""

# Store results
slopes = []
stations1, stations2 = [], []
albedo_trend_modern, albedo_trend_hindcast = [], []
albedo_sig_hindcast, albedo_slope_hindcast, albedo_low_slope, albedo_high_slope = [], [], [], []
t2m_trend_modern, t2m_trend_hindcast = [], []
t2m_sig_hindcast, t2m_slope_hindcast, t2m_low_slope, t2m_high_slope = [], [], [], []
sf_sum_trend_modern, sf_sum_trend_hindcast = [], []
sf_sum_sig_hindcast, sf_sum_slope_hindcast, sf_sum_low_slope, sf_sum_high_slope = [], [], [], []
sf_win_trend_modern, sf_win_trend_hindcast = [], []
sf_win_sig_hindcast, sf_win_slope_hindcast, sf_win_low_slope, sf_win_high_slope = [], [], [], []

for f in range(len(hindcast_files)):
    
    # Read data
    hindcast_df = pd.read_csv(hindcast_files[f], index_col=['datetime'], parse_dates=['datetime'])
    modern_df = pd.read_csv(modern_files[f], index_col=['datetime'], parse_dates=['datetime'])
    modern_df = modern_df[2:]
    
    # Get station name
    s1 = os.path.basename(hindcast_files[f])[:-4]
    stations1.append(s1)
    s2 = os.path.basename(modern_files[f])[:-4]
    stations2.append(s2)
    
    # Run Mann–Kendall tests
    albedo_trend_modern.append(mk.original_test(modern_df['mcd']).trend)
    albedo_trend_hindcast.append(mk.original_test(hindcast_df['y_pred_lin']).trend)
    albedo_sig_hindcast.append(mk.original_test(hindcast_df['y_pred_lin']).p)
    albedo_slope_hindcast.append(mk.original_test(hindcast_df['y_pred_lin']).slope)
    
    slope, intercept, lo_slope, up_slope = theilslopes(hindcast_df['y_pred_lin'], x=hindcast_df['y_pred_lin'].index.year, alpha=0.95)
    albedo_low_slope.append(lo_slope)
    albedo_high_slope.append(up_slope)
    
    t2m_trend_modern.append(mk.original_test(modern_df['t2m']).trend)
    t2m_trend_hindcast.append(mk.original_test(hindcast_df['t2m']).trend)
    t2m_sig_hindcast.append(mk.original_test(hindcast_df['t2m']).p)
    t2m_slope_hindcast.append(mk.original_test(hindcast_df['t2m']).slope)
    
    slope, intercept, lo_slope, up_slope = theilslopes(hindcast_df['t2m'], x=hindcast_df['t2m'].index.year, alpha=0.95)
    t2m_low_slope.append(lo_slope)
    t2m_high_slope.append(up_slope)

    sf_sum_trend_modern.append(mk.original_test(modern_df['sf_summer']).trend)
    sf_sum_trend_hindcast.append(mk.original_test(hindcast_df['sf_summer']).trend)
    sf_sum_sig_hindcast.append(mk.original_test(hindcast_df['sf_summer']).p)
    sf_sum_slope_hindcast.append(mk.original_test(hindcast_df['sf_summer']).slope)
    
    slope, intercept, lo_slope, up_slope = theilslopes(hindcast_df['sf_summer'], x=hindcast_df['sf_summer'].index.year, alpha=0.95)
    sf_sum_low_slope.append(lo_slope)
    sf_sum_high_slope.append(up_slope)
    
    sf_win_trend_modern.append(mk.original_test(modern_df['sf_winter']).trend)
    sf_win_trend_hindcast.append(mk.original_test(hindcast_df['sf_winter']).trend)
    sf_win_sig_hindcast.append(mk.original_test(hindcast_df['sf_winter']).p)
    sf_win_slope_hindcast.append(mk.original_test(hindcast_df['sf_winter']).slope)
    
    slope, intercept, lo_slope, up_slope = theilslopes(hindcast_df['sf_winter'], x=hindcast_df['sf_winter'].index.year, alpha=0.95)
    sf_win_low_slope.append(lo_slope)
    sf_win_high_slope.append(up_slope)

    # Define empty list
    slopes_station = []

    # Loop over all valid 10-year windows
    for i in range(len(hindcast_df) - 9):
        start_date = hindcast_df.index[i]
        end_date = start_date + pd.DateOffset(years=9)
    
        window_df = hindcast_df[(hindcast_df.index >= start_date) & (hindcast_df.index <= end_date)]
    
        # Skip if fewer than 10 data points (e.g., missing years)
        if len(window_df) < 10:
            continue
    
        # Regression
        x = window_df.index.year
        y = window_df['y_pred_lin'].values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
        slopes_station.append(slope)
    
    slopes.append(slopes_station)

# Make DataFrame
slope_df = pd.DataFrame(slopes)
slope_df.index = stations1

#%%
# Sort df1 using that order
slope_df_sorted = slope_df.loc[order]

# Convert to decadal trend
slope_df_sorted = slope_df_sorted * 10

#%%

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

# Example placeholders (replace with your actual data)
station_labels = slope_df_sorted.index
window_labels = hindcast_df.index.year.values

fig, (ax_main, ax_line) = plt.subplots(
    2, 1, figsize=(10, 10), layout='constrained', 
    gridspec_kw={'height_ratios': [4, 1]}, sharex=True
)

# Plot heatmap in the main axis
c = ax_main.imshow(slope_df_sorted, aspect='auto', cmap='coolwarm_r', 
                   interpolation='nearest', vmin=-np.max(np.abs(slope_df_sorted)), 
                   vmax=np.max(np.abs(slope_df_sorted)))

# Colorbar for main heatmap
cbar = fig.colorbar(c, ax=ax_main, pad=-0.09)
cbar.set_label('Albedo trend (decade$^{-1}$)', fontsize=12)
cbar.ax.tick_params(labelsize=12)

# Y-axis ticks for heatmap
ax_main.set_yticks(np.arange(len(station_labels)))
ax_main.set_yticklabels(station_labels, fontsize=12)

# X-axis ticks only on top
xticks = np.arange(0, slope_df_sorted.shape[1], 5)
xtick_labels = [window_labels[i] for i in xticks]
ax_main.set_xticks(xticks)
ax_main.set_xticklabels(xtick_labels, rotation=90, fontsize=12)
ax_main.xaxis.tick_top()
ax_main.tick_params(axis='x', labeltop=True, labelbottom=False)

# Secondary line data
secondary_line = slope_df_sorted.mean()
x = np.arange(len(secondary_line))

# Plot line in separate axis below
ax_line.plot(x, secondary_line, color=c2, alpha=0.8, linewidth=2)
ax_line.set_ylabel('Mean trend (decade$^{-1}$)', fontsize=12)

# Move y-axis label and ticks to the right
ax_line.yaxis.tick_right()
ax_line.yaxis.set_label_position('right')
ax_line.tick_params(labelsize=12)
ax_line.tick_params(axis='x', which='both')
ax_line.set_xticklabels(xtick_labels, rotation=90, fontsize=12)
ax_line.grid(True, which="both", linestyle="--", linewidth=1, zorder=1)
ax_line.axhline(y=0, color='k', alpha=0.7, ls='dashed')
#ax_line.axvline(x=49, color='k', alpha=0.7, ls='dashed')
#ax_line.axvline(x=65, color='k', alpha=0.7, ls='dashed')

plt.savefig(path + 'manuscript/' + 'figX-heatmap-linear.png', dpi=300)

plt.show()


#%%

new_df = pd.DataFrame(slope_df_sorted.mean(), columns=['trend'])
new_df['year'] = hindcast_df.index.year.values[0:76]
new_df.to_csv(path + 'data/trends.csv', index=False)

#%%

"""
Compute trends in air temperature, summer snowfall, and winter snowfall.

"""

df_trend = pd.DataFrame(list(zip(stations1, albedo_trend_modern, albedo_trend_hindcast,
                                 albedo_sig_hindcast, albedo_slope_hindcast,
                                 albedo_low_slope, albedo_high_slope,
                                 t2m_trend_modern, t2m_trend_hindcast,
                                 t2m_sig_hindcast, t2m_slope_hindcast,
                                 t2m_low_slope, t2m_high_slope,
                                 sf_sum_trend_modern, sf_sum_trend_hindcast,
                                 sf_sum_sig_hindcast, sf_sum_slope_hindcast,
                                 sf_sum_low_slope, sf_sum_high_slope,
                                 sf_win_trend_modern, sf_win_trend_hindcast,
                                 sf_win_sig_hindcast, sf_win_slope_hindcast,
                                 sf_win_low_slope, sf_win_high_slope)),
                        columns=['station', 'albedo_trend_modern', 'albedo_trend_hindcast',
                                 'albedo_sig_hindcast', 'albedo_slope_hindcast',
                                 'albedo_low_slope', 'albedo_high_slope',
                                 't2m_trend_modern', 't2m_trend_hindcast',
                                 't2m_sig_hindcast', 't2m_slope_hindcast',
                                 't2m_low_slope', 't2m_high_slope',
                                 'sf_sum_trend_modern', 'sf_sum_trend_hindcast',
                                 'sf_sum_sig_hindcast', 'sf_sum_slope_hindcast',
                                 'sf_sum_low_slope', 'sf_sum_high_slope',
                                 'sf_win_trend_modern', 'sf_win_trend_hindcast',
                                 'sf_win_sig_hindcast', 'sf_win_slope_hindcast',
                                 'sf_win_low_slope', 'sf_win_high_slope'])

df_trend.to_csv(path + 'data/hindcast.csv')

df_trend[['station', 'albedo_trend_modern', 'albedo_trend_hindcast']]
print(np.sum(df_trend['albedo_trend_hindcast'] != 'no trend'))
print(np.sum(df_trend['albedo_trend_hindcast'] == 'decreasing'))
print(np.sum(df_trend['albedo_trend_hindcast'] == 'increasing'))

df_trend[['station', 't2m_trend_modern', 't2m_trend_hindcast']]
print(np.sum(df_trend['t2m_trend_hindcast'] != 'no trend'))
print(np.sum(df_trend['t2m_trend_hindcast'] == 'decreasing'))
print(np.sum(df_trend['t2m_trend_hindcast'] == 'increasing'))

df_trend[['station', 'sf_sum_trend_modern', 'sf_sum_trend_hindcast']]
print(np.sum(df_trend['sf_sum_trend_hindcast'] != 'no trend'))
print(np.sum(df_trend['sf_sum_trend_hindcast'] == 'decreasing'))
print(np.sum(df_trend['sf_sum_trend_hindcast'] == 'increasing'))
 
df_trend[['station', 'sf_win_trend_modern', 'sf_win_trend_hindcast']]
print(np.sum(df_trend['sf_win_trend_hindcast'] != 'no trend'))
print(np.sum(df_trend['sf_win_trend_hindcast'] == 'decreasing'))
print(np.sum(df_trend['sf_win_trend_hindcast'] == 'increasing'))








#%%












