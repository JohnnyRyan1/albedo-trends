#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Trends by station

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

# Define user
user = 'jr555'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/'

# Define files
files = sorted(glob.glob(path + 'data/station/*.csv'))

# Import AWS metadata
aws = pd.read_csv(path + 'data/promice/AWS_sites_metadata.csv')

def rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    return rmse

#%%

station1, diff_before, diff_after  = [], [], []
aws_trend, aws_sig, aws_slope, mcd_trend, mcd_sig, mcd_slope = [], [], [], [], [], []
number = []
diff_trend, diff_sig = [], []
elevation_values, latitude_values = [], []
aws_slope_2012, mcd_slope_2012 = [], []
rmse_mcd, rmse_vnp, rmse_vji = [], [], []
bias_mcd, bias_vnp, bias_vji = [], [], []
rmse_mcd_2018, rmse_vnp_2018 = [], []
mcd_vnp, mcd_vji = [], []
patch_trend, patch_sig = [], []
mcd_trend_2012, vnp_trend_2012 = [], []

for file in files:
    
    # Get station name
    s = os.path.basename(file)[:-4]
    
    # Read file
    df = pd.read_csv(file, index_col=['datetime'], parse_dates=['datetime'])
    df = df[2:]
    
    # Compute differences between AWS and MCD
    mask = np.isfinite(df['aws']) & np.isfinite(df['mcd'])
    if np.sum(np.isfinite(df['aws'])) > 5:
        
        # Difference tests
        diff = df['aws'] - df['mcd']
        diff_before.append(np.nanmean(diff['2002-01-01':'2020-01-01']))
        diff_after.append(np.nanmean(diff['2020-01-01':]))
        
        # RMSE for MCD, VNP, and VJI
        rmse_mcd.append(rmse(df['aws'], df['mcd']))
        rmse_vnp.append(rmse(df['aws'], df['vnp']))
        rmse_vji.append(rmse(df['aws'], df['vji']))
        bias_mcd.append(np.nanmean(df['aws'] - df['mcd']))
        bias_vnp.append(np.nanmean(df['aws'] - df['vnp']))
        bias_vji.append(np.nanmean(df['aws'] - df['vji']))
        
        rmse_mcd_2018.append(rmse(df['aws']['2018-01-01':], df['mcd']['2018-01-01':]))
        rmse_vnp_2018.append(rmse(df['aws']['2018-01-01':], df['vnp']['2018-01-01':]))
        
        mcd_vnp.append(rmse(df['mcd'], df['vnp']))
        mcd_vji.append(rmse(df['mcd'], df['vji']))
        
        # Run Mann–Kendall test
        aws_trend.append(mk.original_test(df['aws']).trend)
        aws_sig.append(mk.original_test(df['aws']).p)
        aws_slope.append(mk.original_test(df['aws']).slope)

        mcd_trend.append(mk.original_test(df['mcd'][mask]).trend) 
        mcd_sig.append(mk.original_test(df['mcd'][mask]).p)
        mcd_slope.append(mk.original_test(df['mcd']).slope)
        
        # Run trend test on patched record
        patch = np.concatenate((df['mcd']['2002-01-01':'2012-01-01'].values, df['vnp']['2012-01-01':].values))
        patch_trend.append(mk.original_test(patch[mask]).trend) 
        patch_sig.append(mk.original_test(patch[mask]).p)
        
        
        if np.sum(df['aws']['2002-01-01':'2013-01-01']) < 5:
            aws_slope_2012.append(np.nan)
            mcd_slope_2012.append(np.nan)
        else:
            aws_slope_2012.append(mk.original_test(df['aws']['2002-01-01':'2013-01-01']).slope)
            mcd_slope_2012.append(mk.original_test(df['mcd']['2002-01-01':'2013-01-01']).slope)

        diff_trend.append(mk.original_test(diff).trend)
        diff_sig.append(mk.original_test(diff).p)
        
        # Append MCD and VNP trends
        mcd_trend_2012.append(mk.original_test(df['mcd']['2012-01-01':]).trend)
        vnp_trend_2012.append(mk.original_test(df['vnp']['2012-01-01':]).trend)
        
        # Append station, elevation, and latitude
        station1.append(s)
        elevation_values.append(aws[aws['site_id'] == s]['altitude_last_valid'].values[0])
        latitude_values.append(aws[aws['site_id'] == s]['latitude_last_valid'].values[0])
        
        # Append count
        number.append(np.sum(mask))
        
        # Define colour map
        c1 = '#E05861'
        c2 = '#616E96'
        c3 = '#F8A557'
        c4 = '#3CBEDD'

        # Set up subplots
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 4), layout='constrained')

        ax1.plot(df['aws'].index.year, df['aws'], color='black', lw=1.5, label='AWS', marker='o', linestyle='-', alpha=0.7)
        ax1.plot(df['aws'].index.year, df['mcd'], color=c1, lw=1.5, label='MCD43A3', marker='o', linestyle='-', alpha=0.7)
        ax1.plot(df['aws'].index.year, df['vnp'], color=c2, lw=1.5, label='VNP43MA3', marker='o', linestyle='-', alpha=0.7)
        ax1.plot(df['aws'].index.year, df['vji'], color=c4, lw=1.5, label='VJI43MA3', marker='o', linestyle='-', alpha=0.7)

        ax1.legend(loc=3, fontsize=12)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.grid(True, which="both", linestyle="--", linewidth=1, zorder=1)
        ax1.set_ylabel("Albedo", fontsize=12)  
        ax1.set_xlim(2001,2025)

        # Add station textbox
        ax1.text(
            0.5, 0.98, s,
            transform=ax1.transAxes,
            fontsize=14,
            verticalalignment='top',
            horizontalalignment='right'
        )

        plt.savefig(path + 'figures/1-products/' + s + '.png')
    

#%%

""" 
RMSE and bias between AWS and MCD43A3.

"""

station_df = pd.DataFrame(list(zip(station1, rmse_mcd, bias_mcd, elevation_values, latitude_values)),
                        columns=['station', 'rmse', 'bias', 'elevation', 'latitude'])

# Save to csv
station_df.to_csv(path + 'data/station-error.csv')

# Report
print('Mean RMSE between AWS and MCD is %.2f' % station_df['rmse'].mean())
print('Mean bias between AWS and MCD is %.2f' % station_df['bias'].mean())

#%%
""" 
Test whether MCD is picking up on the same trends as AWS

"""

# Make DataFrame
stats_df = pd.DataFrame(list(zip(station1, aws_trend, aws_sig, aws_slope, 
                                 mcd_trend, mcd_sig, mcd_slope, aws_slope_2012, mcd_slope_2012)),
                        columns=['station', 'aws_trend', 'aws_sig', 'aws_slope',
                                 'mcd_trend', 'mcd_sig', 'mcd_slope', 
                                 'aws_slope_2012', 'mcd_slope_2012'])

stats_df['agree'] = stats_df['aws_trend'] == stats_df['mcd_trend']

no_trend_agree = stats_df[(stats_df['aws_trend'] == 'no trend') & (stats_df['mcd_trend'] == 'no trend')]
trend = stats_df[stats_df['aws_trend'] != 'no trend']
trend_agree = trend['aws_trend'] == trend['mcd_trend']

# Save to csv
stats_df.to_csv(path + 'data/stats-df.csv')

# Report
print('%.0f sites have no trend according to AWS' % (np.sum(stats_df['aws_trend'] == 'no trend')))
print('%.0f sites where AWS and MCD agree that there is no trend' % len(no_trend_agree))
print('%.0f sites have a trend according to AWS' % len(trend))

print('Percentage of trends that agree with AWS is %.2f' % (np.sum(stats_df['agree']) / len(stats_df)))

print('Mean slope of AWS is %.3f' % (np.mean(stats_df['aws_slope'])))
print('Mean slope of MCD is %.3f' % (np.mean(stats_df['mcd_slope'])))


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

sensor_df = pd.DataFrame(list(zip(station1, rmse_mcd, rmse_vnp, rmse_vji,
                         bias_mcd, bias_vnp, bias_vji, rmse_mcd_2018, rmse_vnp_2018)), 
                         columns=['station','rmse_mcd', 'rmse_vnp', 'rmse_vji',
                                          'bias_mcd', 'bias_vnp', 'bias_vji',
                                          'rmse_mcd_2018', 'rmse_vnp_2018'])

# Report
print('Mean RMSE between AWS and MCD is %.2f' % sensor_df['rmse_mcd'].mean())
print('Mean RMSE between AWS and VNP is %.2f' % sensor_df['rmse_vnp'].mean())
print('Mean RMSE between AWS and VJI is %.2f' % sensor_df['rmse_vji'].mean())
print('Mean bias between AWS and MCD is %.2f' % sensor_df['bias_mcd'].mean())
print('Mean bias between AWS and VNP is %.2f' % sensor_df['bias_vnp'].mean())
print('Mean bias between AWS and VJI is %.2f' % sensor_df['bias_vji'].mean())

print('Mean RMSE between AWS and MCD after 2018 is %.2f' % sensor_df['rmse_mcd_2018'].mean())
print('Mean RMSE between AWS and VNP after 2018 is %.2f' % sensor_df['rmse_vnp_2018'].mean())

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

sensor_diff_df = pd.DataFrame(list(zip(station1, mcd_vnp, mcd_vji)), 
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

print('Mean RMSE between MCD and VNP is %.2f' % sensor_diff_df['mcd_vnp'].mean())
print('Mean RMSE between MCD and VJI is %.2f' % sensor_diff_df['mcd_vji'].mean())


#%%

""" 
Replace 2012-2024 with VNP43MA3 and get the same trends?

"""


# Make DataFrame
patch_df = pd.DataFrame(list(zip(station1, patch_trend, patch_sig)),
                        columns=['station', 'patch_trend', 'patch_sig'])

print('%.0f sites trends are unchanged by replacing MCD with VNP' %np.sum(stats_df['mcd_trend'] == patch_df['patch_trend']))

patch_trend_agree = stats_df['aws_trend'] == patch_df['patch_trend']
mcd_trend_agree = stats_df['aws_trend'] == stats_df['mcd_trend']

# Report
print('%.0f sites have no trend according to AWS' % (np.sum(stats_df['aws_trend'] == 'no trend')))
print('%.0f sites where AWS and MCD agree that there is no trend' % len(no_trend_agree))
print('%.0f sites have a trend according to AWS' % len(trend))

print('Percentage of trends that agree with AWS is %.2f' % (np.sum(stats_df['agree']) / len(stats_df)))


#%%

""" 
Replace 2012-2024 with VNP43MA3 and get the same trends?

"""

















