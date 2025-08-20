#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Figures

"""

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define user
user = 'jr555'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/'

# Read file
station_df = pd.read_csv(path + 'data/station-error-qa.csv')
#linear_df = pd.read_csv(path + 'data/linear-model.csv')
#stats_df = pd.read_csv(path + 'data/stats-df.csv')
#mcd_trends = pd.read_csv(path + 'data/trends.csv')
#hindcast_df = pd.read_csv(path + 'data/hindcast.csv')

#%%

"""
Comparison between AWS, MCD43A3, VNP43MA3, and VJ143MA3 summer albedo at thirty-three sites

"""

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

# Set up subplots
fig, ((ax1, ax2, ax3), 
      (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, 
                                      figsize=(11, 6), sharey=True, sharex=True,
                                      layout='constrained')

ax1.scatter(station_df['aws_mcd'], station_df['mcd'], alpha=0.7, zorder=3, color=c2)
ax1.plot([0,1], [0,1], color='k', alpha=0.5, zorder=2)
ax1.set_ylabel("MCD43A3 (2002-2024)", fontsize=12)  
ax1.set_xlabel("AWS (2002-2024)", fontsize=12)  

ax2.scatter(station_df['aws_vnp'], station_df['vnp'], alpha=0.7, zorder=3, color=c2)
ax2.plot([0,1], [0,1], color='k', alpha=0.5, zorder=2)
ax2.set_ylabel("VNP43MA3 (2012-2024)", fontsize=12)  
ax2.set_xlabel("AWS (2012-2024)", fontsize=12)  

ax3.scatter(station_df['aws_vji'], station_df['vji'], alpha=0.7, zorder=3, color=c2)
ax3.plot([0,1], [0,1], color='k', alpha=0.5, zorder=2)
ax3.set_ylabel("VJ143MA3 (2018-2024)", fontsize=12)  
ax3.set_xlabel("AWS (2018-2024)", fontsize=12)  

ax4.scatter(station_df['mcd'], station_df['vnp'], alpha=0.7, zorder=3, color=c2)
ax4.plot([0,1], [0,1], color='k', alpha=0.5, zorder=2)
ax4.set_ylabel("VNP43MA3 (2012-2024)", fontsize=12)  
ax4.set_xlabel("MCD43A3 (2012-2024)", fontsize=12)  

ax5.scatter(station_df['mcd'], station_df['vji'], alpha=0.7, zorder=3, color=c2)
ax5.plot([0,1], [0,1], color='k', alpha=0.5, zorder=2)
ax5.set_ylabel("VJ143MA3 (2018-2024)", fontsize=12)  
ax5.set_xlabel("MCD43A3 (2018-2024)", fontsize=12)  

ax6.scatter(station_df['vnp'], station_df['vji'], alpha=0.7, zorder=3, color=c2)
ax6.plot([0,1], [0,1], color='k', alpha=0.5, zorder=2)
ax6.set_ylabel("VJ143MA3 (2018-2024)", fontsize=12)  
ax6.set_xlabel("VNP43MA3 (2018-2024)", fontsize=12)  

# Add text box with regression stats
textstr = '\n'.join((
    f"RMSE: {station_df['rmse_aws_mcd'].mean():.3f}",
    f"Bias: {station_df['bias_aws_mcd'].mean():.3f}",
))
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
ax1.text(0.64, 0.18, textstr, transform=ax1.transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

textstr = '\n'.join((
    f"RMSE: {station_df['rmse_aws_vnp'].mean():.3f}",
    f"Bias: {station_df['bias_aws_vnp'].mean():.3f}",
))
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
ax2.text(0.64, 0.18, textstr, transform=ax2.transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

textstr = '\n'.join((
    f"RMSE: {station_df['rmse_aws_vji'].mean():.3f}",
    f"Bias: {station_df['bias_aws_vji'].mean():.3f}",
))
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
ax3.text(0.64, 0.18, textstr, transform=ax3.transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

textstr = '\n'.join((
    f"RMSE: {station_df['rmse_mcd_vnp'].mean():.3f}",
    f"Bias: {station_df['bias_mcd_vnp'].mean():.3f}",
))
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
ax4.text(0.64, 0.18, textstr, transform=ax4.transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

textstr = '\n'.join((
    f"RMSE: {station_df['rmse_mcd_vji'].mean():.3f}",
    f"Bias: {station_df['bias_mcd_vji'].mean():.3f}",
))
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
ax5.text(0.64, 0.18, textstr, transform=ax5.transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

textstr = '\n'.join((
    f"RMSE: {station_df['rmse_vnp_vji'].mean():.3f}",
    f"Bias: {station_df['bias_vnp_vji'].mean():.3f}",
))
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
ax6.text(0.64, 0.18, textstr, transform=ax6.transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_ylim(0.2, 0.9)
    ax.set_xlim(0.2, 0.9)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=1, zorder=1)

ax1.text(0.03, 0.98, 'a', transform=ax1.transAxes,
         fontsize=20, va='top', ha='left')
ax2.text(0.03, 0.98, 'b', transform=ax2.transAxes,
         fontsize=20, va='top', ha='left')
ax3.text(0.03, 0.98, 'c', transform=ax3.transAxes,
         fontsize=20, va='top', ha='left')
ax4.text(0.03, 0.98, 'd', transform=ax4.transAxes,
         fontsize=20, va='top', ha='left')
ax5.text(0.03, 0.98, 'e', transform=ax5.transAxes,
         fontsize=20, va='top', ha='left')
ax6.text(0.03, 0.98, 'f', transform=ax6.transAxes,
         fontsize=20, va='top', ha='left')

plt.savefig(path + 'manuscript/figX-scatterplots-qa.png', dpi=300)


#%%













