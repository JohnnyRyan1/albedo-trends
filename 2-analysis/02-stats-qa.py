#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Trends by station (QA filtered)

"""

# Import packages
import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.stats import ttest_rel
import glob
import os
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy.stats import theilslopes

# Define user
user = 'jr555'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/'

# Define files
files = sorted(glob.glob(path + 'data/station-qa/*.csv'))

# Import AWS metadata
aws = pd.read_csv(path + 'data/promice/AWS_sites_metadata.csv')

def rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    return rmse

#%%

station1, number, elevation_values, latitude_values, diff_before, diff_after  = [], [], [], [], [], []
diff_trend, diff_sig = [], []
rmse_aws_mcd, rmse_aws_vnp, rmse_aws_vji = [], [], []
bias_aws_mcd, bias_aws_vnp, bias_aws_vji = [], [], []
rmse_mcd_vnp, rmse_mcd_vji, rmse_vnp_vji = [], [], []
bias_mcd_vnp, bias_mcd_vji, bias_vnp_vji = [], [], []
albedo_aws_mcd, albedo_mcd, albedo_aws_vnp, albedo_vnp, albedo_aws_vji, albedo_vji = [], [], [], [], [], []

mcd_trend_2002, mcd_slope_2002, mcd_sig_2002 = [], [], []
mcd_low_slope_2002, mcd_high_slope_2002 = [], []
mcd_trend_2012, mcd_slope_2012, mcd_sig_2012 = [], [], []
mcd_low_slope_2012, mcd_high_slope_2012 = [], []
mcd_trend_2018, mcd_slope_2018, mcd_sig_2018 = [], [], []

vnp_trend_2012, vnp_slope_2012, vnp_sig_2012 = [], [], []
vnp_low_slope_2012, vnp_high_slope_2012 = [], []
vnp_trend_2018, vnp_slope_2018, vnp_sig_2018 = [], [], []

patch_trend, patch_sig, patch_slope = [], [], []

for file in files:
    
    # Get station name
    s = os.path.basename(file)[:-4]
    
    # Read file
    df = pd.read_csv(file, index_col=['datetime'], parse_dates=['datetime'])
    df = df[2:]
    
    # Compute differences between AWS and MCD
    mask_mcd = np.isfinite(df['aws']) & np.isfinite(df['mcd'])
    mask_vnp = np.isfinite(df['aws']) & np.isfinite(df['vnp'])
    mask_vji = np.isfinite(df['aws']) & np.isfinite(df['vji'])

    if np.sum(np.isfinite(df['aws'])) > 5:
        
        # Difference tests
        diff = df['aws'] - df['mcd']
        diff_before.append(np.nanmean(diff['2002-01-01':'2020-01-01']))
        diff_after.append(np.nanmean(diff['2020-01-01':]))
        diff_trend.append(mk.original_test(diff).trend)
        diff_sig.append(mk.original_test(diff).p)
        
        # RMSE and bias for MCD, VNP, and VJ1
        rmse_aws_mcd.append(rmse(df['aws'], df['mcd']))
        rmse_aws_vnp.append(rmse(df['aws'], df['vnp']))
        rmse_aws_vji.append(rmse(df['aws'], df['vji']))
        bias_aws_mcd.append(np.nanmean(df['aws'] - df['mcd']))
        bias_aws_vnp.append(np.nanmean(df['aws'] - df['vnp']))
        bias_aws_vji.append(np.nanmean(df['aws'] - df['vji']))
        rmse_mcd_vnp.append(rmse(df['mcd'], df['vnp']))
        rmse_mcd_vji.append(rmse(df['mcd'], df['vji']))
        rmse_vnp_vji.append(rmse(df['vnp'], df['vji']))
        bias_mcd_vnp.append(np.nanmean(df['mcd'] - df['vnp']))
        bias_mcd_vji.append(np.nanmean(df['mcd'] - df['vji']))
        bias_vnp_vji.append(np.nanmean(df['vnp'] - df['vji']))
        
        albedo_aws_mcd.append(df['aws'][mask_mcd].mean())
        albedo_mcd.append(df['mcd'][mask_mcd].mean())
        albedo_aws_vnp.append(df['aws'][mask_vnp].mean())
        albedo_vnp.append(df['vnp'][mask_vnp].mean())
        albedo_aws_vji.append(df['aws'][mask_vji].mean())
        albedo_vji.append(df['vji'][mask_vji].mean())

        # Trends        
        mcd_trend_2002.append(mk.original_test(df['mcd']).trend) 
        mcd_sig_2002.append(mk.original_test(df['mcd']).p)
        mcd_slope_2002.append(mk.original_test(df['mcd']).slope)
        
        slope, intercept, lo_slope, up_slope = theilslopes(df['mcd'], x=df['mcd'].index.year, alpha=0.95)
        mcd_low_slope_2002.append(lo_slope)
        mcd_high_slope_2002.append(up_slope)
        
        mcd_trend_2012.append(mk.original_test(df['mcd']['2012-01-01':]).trend) 
        mcd_sig_2012.append(mk.original_test(df['mcd']['2012-01-01':]).p)
        mcd_slope_2012.append(mk.original_test(df['mcd']['2012-01-01':]).slope)
        
        slope, intercept, lo_slope, up_slope = theilslopes(df['mcd']['2012-01-01':], x=df['mcd']['2012-01-01':].index.year, alpha=0.95)
        mcd_low_slope_2012.append(lo_slope)
        mcd_high_slope_2012.append(up_slope)
        
        if np.isfinite(df['vnp']).sum() > 2:
        
            vnp_trend_2012.append(mk.original_test(df['vnp']).trend) 
            vnp_sig_2012.append(mk.original_test(df['vnp']).p)
            vnp_slope_2012.append(mk.original_test(df['vnp']).slope)
            
            slope, intercept, lo_slope, up_slope = theilslopes(df['vnp']['2012-01-01':], x=df['vnp']['2012-01-01':].index.year, alpha=0.95)
            vnp_low_slope_2012.append(lo_slope)
            vnp_high_slope_2012.append(up_slope)
        else:
            vnp_trend_2012.append(np.nan) 
            vnp_sig_2012.append(np.nan)
            vnp_slope_2012.append(np.nan)
            vnp_low_slope_2012.append(slope)
            vnp_high_slope_2012.append(slope)

        # Run trend test on patched record
        patch = np.concatenate((df['mcd']['2002-01-01':'2012-01-01'].values, df['vnp']['2012-01-01':].values))
        patch_trend.append(mk.original_test(patch).trend) 
        patch_sig.append(mk.original_test(patch).p)
        patch_slope.append(mk.original_test(patch).slope)
        
        # Append station, elevation, and latitude
        station1.append(s)
        elevation_values.append(aws[aws['site_id'] == s]['altitude_last_valid'].values[0])
        latitude_values.append(aws[aws['site_id'] == s]['latitude_last_valid'].values[0])
        
        # Append count
        number.append(np.sum(mask_mcd))
        
# =============================================================================
#         # Define colour map
#         c1 = '#E05861'
#         c2 = '#616E96'
#         c3 = '#F8A557'
#         c4 = '#3CBEDD'
# 
#         # Set up subplots
#         fig, ax1 = plt.subplots(1, 1, figsize=(7, 4), layout='constrained')
# 
#         ax1.plot(df['aws'].index.year, df['aws'], color='black', lw=1.5, label='AWS', marker='o', linestyle='-', alpha=0.7)
#         ax1.plot(df['aws'].index.year, df['mcd'], color=c1, lw=1.5, label='MCD43A3', marker='o', linestyle='-', alpha=0.7)
#         ax1.plot(df['aws'].index.year, df['vnp'], color=c2, lw=1.5, label='VNP43MA3', marker='o', linestyle='-', alpha=0.7)
#         ax1.plot(df['aws'].index.year, df['vji'], color=c4, lw=1.5, label='VJI43MA3', marker='o', linestyle='-', alpha=0.7)
# 
#         ax1.legend(loc=3, fontsize=12)
#         ax1.tick_params(axis='both', which='major', labelsize=12)
#         ax1.grid(True, which="both", linestyle="--", linewidth=1, zorder=1)
#         ax1.set_ylabel("Albedo", fontsize=12)  
#         ax1.set_xlim(2001,2025)
# 
#         # Add station textbox
#         ax1.text(
#             0.5, 0.98, s,
#             transform=ax1.transAxes,
#             fontsize=14,
#             verticalalignment='top',
#             horizontalalignment='right'
#         )
# 
#         plt.savefig(path + 'figures/1-products/' + s + '.png')
# =============================================================================
        
# =============================================================================
#         # Define colour map
#         c1 = '#E05861'
#         c2 = '#616E96'
#         c3 = '#F8A557'
#         c4 = '#3CBEDD'
# 
#         # Set up subplots
#         fig, ax1 = plt.subplots(1, 1, figsize=(7, 4), layout='constrained')
#         ax1.plot(df['mcd'].index.year, patch, color=c1, lw=1.5, label='Patch', marker='o', linestyle='--', alpha=0.7)
#         ax1.plot(df['mcd'].index.year, df['mcd'], color='black', lw=1.5, label='MCD', marker='o', linestyle='-', alpha=0.7)
# 
#         ax1.legend(loc=3, fontsize=12)
#         ax1.tick_params(axis='both', which='major', labelsize=12)
#         ax1.grid(True, which="both", linestyle="--", linewidth=1, zorder=1)
#         ax1.set_ylabel("Albedo", fontsize=12)  
#         ax1.set_xlim(2001,2025)
# 
#         # Add station textbox
#         ax1.text(
#             0.5, 0.98, s,
#             transform=ax1.transAxes,
#             fontsize=14,
#             verticalalignment='top',
#             horizontalalignment='right'
#         )
# 
#         plt.savefig(path + 'figures/5-patched/' + s + '.png')
# =============================================================================

#%%

""" 
RMSE and bias between AWS and MCD43A3.

"""

station_df = pd.DataFrame(list(zip(station1, albedo_aws_mcd, albedo_aws_vnp, albedo_aws_vji,
                                   albedo_mcd, albedo_vnp, albedo_vji,
                                   rmse_aws_mcd, bias_aws_mcd, rmse_aws_vnp, 
                                   bias_aws_vnp, rmse_aws_vji, bias_aws_vji, 
                                   rmse_mcd_vnp, rmse_mcd_vji, rmse_vnp_vji,
                                   bias_mcd_vnp, bias_mcd_vji, bias_vnp_vji,
                                   elevation_values, latitude_values)),
                        columns=['station', 'aws_mcd', 'aws_vnp', 'aws_vji', 
                                 'mcd', 'vnp', 'vji', 
                                 'rmse_aws_mcd', 'bias_aws_mcd', 'rmse_aws_vnp', 
                                 'bias_aws_vnp', 'rmse_aws_vji', 'bias_aws_vji',
                                 'rmse_mcd_vnp', 'rmse_mcd_vji', 'rmse_vnp_vji',
                                 'bias_mcd_vnp', 'bias_mcd_vji', 'bias_vnp_vji',
                                 'elevation', 'latitude'])

# Save to csv
station_df.to_csv(path + 'data/station-error-qa.csv')

# Report
print('Mean RMSE between AWS and MCD is %.2f' % station_df['rmse_aws_mcd'].mean())
print('Mean bias between AWS and MCD is %.2f' % station_df['bias_aws_mcd'].mean())

#%%

""" 
Trend in MCD for the 2002-2024 period.

"""

# Make DataFrame
trend_df = pd.DataFrame(list(zip(station1, 
                                 mcd_trend_2002, mcd_sig_2002, mcd_slope_2002, 
                                 mcd_low_slope_2002, mcd_high_slope_2002,
                                 mcd_trend_2012, mcd_sig_2012, mcd_slope_2012, 
                                 mcd_low_slope_2012, mcd_high_slope_2012,
                                 vnp_trend_2012, vnp_sig_2012, vnp_slope_2012, 
                                 vnp_low_slope_2012, vnp_high_slope_2012,
                                 patch_trend, patch_sig, patch_slope,
                                 elevation_values)),
                        columns=['station', 
                                 'mcd_trend_2002', 'mcd_sig_2002', 'mcd_slope_2002', 
                                 'mcd_low_slope_2002', 'mcd_high_slope_2002',
                                 'mcd_trend_2012', 'mcd_sig_2012', 'mcd_slope_2012', 
                                 'mcd_low_slope_2012', 'mcd_high_slope_2012',
                                 'vnp_trend_2012', 'vnp_sig_2012', 'vnp_slope_2012', 
                                 'vnp_low_slope_2012', 'vnp_high_slope_2012',
                                 'patch_trend', 'patch_sig', 'patch_slope',
                                 'elevation'])

trend_df.to_csv(path + 'data/trends-qa.csv', index=False)


#%%

"""

Is the difference between MCD and AWS growing through time?

"""

# Make DataFrame
diff_df = pd.DataFrame(list(zip(station1, diff_trend, diff_sig, number)),
                        columns=['station', 'diff_trend', 'diff_sig', 'count'])

diff_no_trend = diff_df[diff_df['diff_trend'] == 'no trend']
diff_increasing = diff_df[diff_df['diff_trend'] == 'increasing']
diff_decreasing = diff_df[diff_df['diff_trend'] == 'decreasing']

print('%.0f sites where difference between AWS and MCD is not changing' % (len(diff_no_trend)))
print('%.0f sites where difference between AWS and MCD is increasing' % (len(diff_increasing)))
print('%.0f sites where difference between AWS and MCD is decreasing' % (len(diff_decreasing)))

#%%

""" 
Test whether orbital drift is increasing the difference between AWS and MCD

"""
mask = np.isfinite(np.array(diff_before)) & np.isfinite(np.array(diff_after))
a1, a2 = np.array(diff_before)[mask], np.array(diff_after)[mask]

t_stat, p_val = ttest_rel(a1, a2)

print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.3g}")

#%%

""" 
Is RMSE and bias correlated with latitude or elevation?

"""

slope1, intercept1, rvalue1, pvalue1, std_err1 = linregress(station_df['rmse'], station_df['elevation'])
slope2, intercept2, rvalue2, pvalue2, std_err2 = linregress(station_df['bias'], station_df['elevation'])
slope3, intercept3, rvalue3, pvalue3, std_err3 = linregress(station_df['rmse'], station_df['latitude'])
slope4, intercept4, rvalue4, pvalue4, std_err4 = linregress(station_df['bias'], station_df['latitude'])

#%%

""" 
Report mean RMSE and bias between AWS and MCD43A3, VNP43MA3, and VJI43MA3

"""

sensor_df = pd.DataFrame(list(zip(station1, rmse_aws_mcd, rmse_aws_vnp, rmse_aws_vji,
                         bias_aws_mcd, bias_aws_vnp, bias_aws_vji)), 
                         columns=['station','rmse_mcd', 'rmse_vnp', 'rmse_vji',
                                          'bias_mcd', 'bias_vnp', 'bias_vji'])

# Report
print('Mean RMSE between AWS and MCD is %.2f' % sensor_df['rmse_mcd'].mean())
print('Mean RMSE between AWS and VNP is %.2f' % sensor_df['rmse_vnp'].mean())
print('Mean RMSE between AWS and VJI is %.2f' % sensor_df['rmse_vji'].mean())
print('Mean bias between AWS and MCD is %.2f' % sensor_df['bias_mcd'].mean())
print('Mean bias between AWS and VNP is %.2f' % sensor_df['bias_vnp'].mean())
print('Mean bias between AWS and VJI is %.2f' % sensor_df['bias_vji'].mean())

# Set up subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')

# RMSE boxplot
sensor_df[['rmse_mcd', 'rmse_vnp', 'rmse_vji']].boxplot(ax=axes[0], showfliers=False)
axes[0].set_ylabel('RMSE', fontsize=12)
axes[0].set_xticklabels(['MCD43A3', 'VNP43MA3', 'VJI43MA3'], fontsize=12)
axes[0].grid(True)

# Bias boxplot
sensor_df[['bias_mcd', 'bias_vnp', 'bias_vji']].boxplot(ax=axes[1], showfliers=False)
axes[1].set_ylabel('Bias', fontsize=12)
axes[1].set_xticklabels(['MCD43A3', 'VNP43MA3', 'VJI43MA3'], fontsize=12)
axes[1].grid(True)

#%%

""" 
Report mean RMSE between MCD43A3, VNP43MA3, and VJI43MA3

"""

sensor_diff_df = pd.DataFrame(list(zip(station1, rmse_mcd_vnp, rmse_mcd_vji)), 
                         columns=['station','mcd_vnp', 'mcd_vji'])


print('Mean RMSE between MCD and VNP is %.2f' % sensor_diff_df['mcd_vnp'].mean())
print('Mean RMSE between MCD and VJI is %.2f' % sensor_diff_df['mcd_vji'].mean())


#%%
""" 
Do MCD43A3 and VNP43MA3 agree on trends for the 2012-2024 period?

"""

sensor_trend_df = pd.DataFrame(list(zip(station1, mcd_trend_2012, vnp_trend_2012)), 
                         columns=['station','mcd', 'vnp'])
sensor_trend_df['agree'] = sensor_trend_df['mcd'] == sensor_trend_df['vnp']



#%%

""" 
Replace 2012-2024 with VNP43MA3 and get the same trends?

"""


# Make DataFrame
patch_df = pd.DataFrame(list(zip(station1, mcd_trend_2002, patch_trend)),
                        columns=['station', 'mcd_trend', 'patch_trend'])

print('%.0f sites trends are unchanged by replacing MCD with VNP' %np.sum(trend_df['mcd_trend_2002'] == patch_df['patch_trend']))


#%%


















