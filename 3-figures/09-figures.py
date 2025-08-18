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
station_df = pd.read_csv(path + 'data/station-error.csv')
linear_df = pd.read_csv(path + 'data/linear-model.csv')
stats_df = pd.read_csv(path + 'data/stats-df.csv')
mcd_trends = pd.read_csv(path + 'data/trends.csv')
hindcast_df = pd.read_csv(path + 'data/hindcast.csv')
aws_meta = pd.read_csv(path +'data/promice/AWS_sites_metadata_updated.csv')

#%%

"""
Comparison between AWS, MCD43A3, VNP43MA3, and VJ143MA3 summer albedo at thirty-three sites

"""

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

station_df['zone'] = aws_meta['zone']
zone_colors = {'Accumulation': c2, 'Ablation': c1}
colors = station_df['zone'].map(zone_colors)

# Set up subplots
fig, ((ax1, ax2, ax3), 
      (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, 
                                      figsize=(11, 6), sharey=True, sharex=True,
                                      layout='constrained')

ax1.scatter(station_df['aws_mcd'], station_df['mcd'], alpha=0.7, zorder=3, color=colors)
ax1.plot([0,1], [0,1], color='k', alpha=0.5, zorder=2)
ax1.set_ylabel("MCD43A3 (2002-2024)", fontsize=12)  
ax1.set_xlabel("AWS (2002-2024)", fontsize=12)  

ax2.scatter(station_df['aws_vnp'], station_df['vnp'], alpha=0.7, zorder=3, color=colors)
ax2.plot([0,1], [0,1], color='k', alpha=0.5, zorder=2)
ax2.set_ylabel("VNP43MA3 (2012-2024)", fontsize=12)  
ax2.set_xlabel("AWS (2012-2024)", fontsize=12)  

ax3.scatter(station_df['aws_vji'], station_df['vji'], alpha=0.7, zorder=3, color=colors)
ax3.plot([0,1], [0,1], color='k', alpha=0.5, zorder=2)
ax3.set_ylabel("VJ143MA3 (2018-2024)", fontsize=12)  
ax3.set_xlabel("AWS (2018-2024)", fontsize=12)  

ax4.scatter(station_df['mcd'], station_df['vnp'], alpha=0.7, zorder=3, color=colors)
ax4.plot([0,1], [0,1], color='k', alpha=0.5, zorder=2)
ax4.set_ylabel("VNP43MA3 (2012-2024)", fontsize=12)  
ax4.set_xlabel("MCD43A3 (2012-2024)", fontsize=12)  

ax5.scatter(station_df['mcd'], station_df['vji'], alpha=0.7, zorder=3, color=colors)
ax5.plot([0,1], [0,1], color='k', alpha=0.5, zorder=2)
ax5.set_ylabel("VJ143MA3 (2018-2024)", fontsize=12)  
ax5.set_xlabel("MCD43A3 (2018-2024)", fontsize=12)  

ax6.scatter(station_df['vnp'], station_df['vji'], alpha=0.7, zorder=3, color=colors)
ax6.plot([0,1], [0,1], color='k', alpha=0.5, zorder=2)
ax6.set_ylabel("VJ143MA3 (2018-2024)", fontsize=12)  
ax6.set_xlabel("VNP43MA3 (2018-2024)", fontsize=12)  

# Add text box with regression stats
textstr = '\n'.join((
    f"RMSE: {station_df['rmse_aws_mcd'].mean():.3f}",
    f"Bias: {station_df['bias_aws_mcd'].mean():.3f}",
))
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
ax1.text(0.64, 0.4, textstr, transform=ax1.transAxes, fontsize=12,
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

custom_legend = [
    Line2D([0], [0], marker='o', color='w', label='Accumulation zone',
           markerfacecolor=zone_colors['Accumulation'], markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Ablation zone',
           markerfacecolor=zone_colors['Ablation'], markersize=8)
]
ax1.legend(handles=custom_legend, loc='lower right', fontsize=11)

plt.savefig(path + 'manuscript/figX-scatterplots.png', dpi=300)


#%%

"""
Ranked lollipop chart for MCD and MCD/VNP for 2002-2024.

"""

# Sort by slope
df_sorted = mcd_trends.sort_values(by="elevation").reset_index(drop=True)

# Create significance flag
df_sorted["sig_flag"] = df_sorted["mcd_sig_2002"].apply(lambda p: "Significant" if p < 0.05 else "Not Significant")
df_sorted["patch_flag"] = df_sorted["patch_sig"].apply(lambda p: "Significant" if p < 0.05 else "Not Significant")

# Main figure and axes
fig, (ax) = plt.subplots(
    1, 1,
    figsize=(8, 4),
    layout='constrained')

# Colors for points based on significance
colors1 = df_sorted["sig_flag"].map({"Significant": c1, "Not Significant": c2})
colors2 = df_sorted["patch_flag"].map({"Significant": c1, "Not Significant": c2})

# --- Main plot ---
# Confidence intervals
ax.vlines(
    x=df_sorted.index,
    ymin=df_sorted["mcd_low_slope_2002"],
    ymax=df_sorted["mcd_high_slope_2002"],
    color="k",
    linewidth=1.5,
    zorder=3,
    alpha=0.5
)

# Slope points
ax.scatter(df_sorted.index, df_sorted["mcd_slope_2002"], c=colors1, s=60, zorder=3)
ax.scatter(df_sorted.index, df_sorted["patch_slope"], 
           c=colors2, s=60, zorder=3, marker='x')

# Zero line
ax.axhline(0, color="black", linewidth=1.5, linestyle="--", zorder=2)

# Axis formatting
ax.set_xticks(df_sorted.index)
ax.set_xticklabels(df_sorted["station"], rotation=90)
ax.xaxis.tick_top()
ax.tick_params(axis='x', labeltop=True, labelbottom=False)
ax.tick_params(axis='both', which='major', labelsize=11)
ax.set_ylabel("MCD43A3 albedo trend 2002-2024 (yr$^{-1}$)", fontsize=12)
ax.set_ylim(-0.01, 0.012)
ax.grid(True, which="both", linestyle="--", linewidth=1, zorder=1)


# --- Legend ---
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='MCD43A3 significant (p < 0.05)',
           markerfacecolor=c1, markersize=8),
    Line2D([0], [0], marker='o', color='w', label='MCD43A3 not significant (p > 0.05)',
           markerfacecolor=c2, markersize=8),
    Line2D([0], [0], marker='x', color=c1, label='MCD43A3/VNP43MA3 p < 0.05',
           markersize=8, linestyle='None'),
    Line2D([0], [0], marker='x', color=c2, label='MCD43A3/VNP43MA3 p < 0.05',
           markersize=8, linestyle='None')
]

ax.legend(handles=legend_elements, loc=4, fontsize=11)

plt.savefig(path + 'manuscript/figX-mcd-2002-lollipop.png', dpi=300)


#%%

"""
Ranked lollipop chart for MCD and VNP for 2012-2024.

"""

# Sort by slope
df_sorted = mcd_trends.sort_values(by="elevation").reset_index(drop=True)

# Create significance flag
df_sorted["sig_flag"] = df_sorted["mcd_sig_2012"].apply(lambda p: "Significant" if p < 0.05 else "Not Significant")
df_sorted["vnp_flag"] = df_sorted["vnp_sig_2012"].apply(lambda p: "Significant" if p < 0.05 else "Not Significant")

# Main figure and axes
fig, (ax) = plt.subplots(
    1, 1,
    figsize=(8, 4),
    layout='constrained')

# Colors for points based on significance
colors1 = df_sorted["sig_flag"].map({"Significant": c1, "Not Significant": c2})
colors2 = df_sorted["vnp_flag"].map({"Significant": c1, "Not Significant": c2})

# --- Main plot ---
# Confidence intervals
ax.vlines(
    x=df_sorted.index,
    ymin=df_sorted["mcd_low_slope_2012"],
    ymax=df_sorted["mcd_high_slope_2012"],
    color="k",
    linewidth=1.5,
    zorder=3,
    alpha=0.5
)

# Slope points
ax.scatter(df_sorted.index, df_sorted["mcd_slope_2012"], c=colors1, s=60, zorder=3)
ax.scatter(df_sorted.index, df_sorted["vnp_slope_2012"], c=colors2, s=60, zorder=3,
           marker='x')



# Zero line
ax.axhline(0, color="black", linewidth=1.5, linestyle="--", zorder=2)

# Axis formatting
ax.set_xticks(df_sorted.index)
ax.set_xticklabels(df_sorted["station"], rotation=90)
ax.xaxis.tick_top()
ax.tick_params(axis='x', labeltop=True, labelbottom=False)
ax.tick_params(axis='both', which='major', labelsize=11)
ax.set_ylabel("MCD43A3 albedo trend 2002-2024 (yr$^{-1}$)", fontsize=12)
ax.set_ylim(-0.02, 0.02)
ax.grid(True, which="both", linestyle="--", linewidth=1, zorder=1)


# --- Legend ---
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='MCD43A3 (p < 0.05)',
           markerfacecolor=c1, markersize=8),
    Line2D([0], [0], marker='o', color='w', label='MCD43A3 (p > 0.05)',
           markerfacecolor=c2, markersize=8),
    Line2D([0], [0], marker='x', color=c1, label='VNP43MA3 (p < 0.05)',
           markersize=8, linestyle='None'),
    Line2D([0], [0], marker='x', color=c2, label='VNP43MA3 (p < 0.05)',
           markersize=8, linestyle='None')
]

ax.legend(handles=legend_elements, loc=4, fontsize=11)

plt.savefig(path + 'manuscript/figX-mcd-2012-lollipop.png', dpi=300)

#%%

"""
Comparison between AWS, MCD43A3, mcd43MA3, and VJ143MA3 summer albedo at thirty-three sites

"""

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

# Set up subplots
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), layout='constrained')

ax1.scatter(linear_df['observed'], linear_df['predicted'], alpha=0.7, color=c2, zorder=3)
ax1.plot([0,1], [0,1], color='k', alpha=0.5, zorder=2)
ax1.set_ylabel("Predicted summer albedo (2002-2024)", fontsize=12)  
ax1.set_xlabel("Observed summer albedo (2002-2024)", fontsize=12)  

ax1.set_ylim(0.2, 0.9)
ax1.set_xlim(0.2, 0.9)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True, which="both", linestyle="--", linewidth=1, zorder=1)

# Add text box with regression stats
textstr = '\n'.join((
    f"RMSE: {linear_df['rmse'].mean():.3f}",
))
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
ax1.text(0.72, 0.06, textstr, transform=ax1.transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

plt.savefig(path + 'manuscript/figX-linear-model.png', dpi=300)

#%%

"""
Ranked lollipop chart for hindast for 1941-2024.

"""

# Define order
order = df_sorted['station'].tolist()

# Sort df1 using that order
df_sorted = hindcast_df.set_index("station").loc[order].reset_index()

# Create significance flag
df_sorted["sig_flag"] = df_sorted["albedo_sig_hindcast"].apply(lambda p: "Significant" if p < 0.05 else "Not Significant")

# Main figure and axes
fig, (ax) = plt.subplots(
    1, 1,
    figsize=(8, 4),
    layout='constrained')

# Colors for points based on significance
colors1 = df_sorted["sig_flag"].map({"Significant": c1, "Not Significant": c2})

# --- Main plot ---
# Confidence intervals
ax.vlines(
    x=df_sorted.index,
    ymin=df_sorted["albedo_low_slope"],
    ymax=df_sorted["albedo_high_slope"],
    color="k",
    linewidth=1.5,
    zorder=3,
    alpha=0.5
)

# Slope points
ax.scatter(df_sorted.index, df_sorted["albedo_slope_hindcast"], c=colors1, s=60, zorder=3)

# Zero line
ax.axhline(0, color="black", linewidth=1.5, linestyle="--", zorder=2)

# Axis formatting
ax.set_xticks(df_sorted.index)
ax.set_xticklabels(df_sorted["station"], rotation=90)
ax.xaxis.tick_top()
ax.tick_params(axis='x', labeltop=True, labelbottom=False)
ax.tick_params(axis='both', which='major', labelsize=11)
ax.set_ylabel("MCD43A3 albedo trend 1941-2024 (yr$^{-1}$)", fontsize=12)
#ax.set_ylim(-0.02, 0.02)
ax.grid(True, which="both", linestyle="--", linewidth=1, zorder=1)


# --- Legend ---
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Significant (p < 0.05)',
           markerfacecolor=c1, markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Not significant (p > 0.05)',
           markerfacecolor=c2, markersize=8),
]

ax.legend(handles=legend_elements, loc=4, fontsize=11)

plt.savefig(path + 'manuscript/figX-mcd-1941-lollipop.png', dpi=300)



#%%













