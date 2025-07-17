"""
Created on Thu Apr  3 17:06:59 2025

This script produces timeseries of albedo with criteria of regions and elevations and has for loop to do all graphs at once
has modifications of grey area and adjusted range

@author: henrykuemmel
"""
import os
import rasterio
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from scipy.stats import linregress
import calendar

#paths
base_directory = "/Volumes/TOSHIBA/Ryan2/TIFS"
mask_path = "/Volumes/TOSHIBA/Ryan2/1km-ISMIP6-GIMP.nc"
regions_path = "/Volumes/TOSHIBA/Ryan2/temp_albedo_summer_climatologies.nc"
output_dir = "/Volumes/TOSHIBA/Ryan2/TimesSeries3"
os.makedirs(output_dir, exist_ok=True)
    
#load mask and elevation and regions data
ismip_1km = xr.open_dataset(mask_path)
mask = ismip_1km['GIMP'].values
elevation = ismip_1km['SRF'].values
regions_file = xr.open_dataset(regions_path)
regions = regions_file['regions'].values

#name regions 1–8
region_names = {
    1: "North",
    2: "Northeast",
    3: "East",
    4: "Southeast",
    5: "South",
    6: "Southwest",
    7: "West",
    8: "Northwest"
}
#determine maximum y-range across all region/elevation/month combos
max_range = 0

for month_num in range(4, 11):  
    target_month = f"{month_num:02d}"

    for elevation_above in [True, False]:
        for target_region in range(1, 9):

            temp_albedos = []

            for root, _, files in os.walk(base_directory):
                for filename in sorted(files):
                    if filename.endswith(".tif") and "_qa" not in filename:
                        parts = filename.split("-")
                        if len(parts) == 3 and parts[2][:2] == target_month:
                            file_path = os.path.join(root, filename)
                            with rasterio.open(file_path) as dataset:
                                albedo_data = dataset.read(1)
                                albedo_data = np.where(albedo_data == dataset.nodata, np.nan, albedo_data)

                                if mask.shape != albedo_data.shape:
                                    resampled_mask = np.zeros_like(albedo_data, dtype=np.float32)
                                    resampled_regions = np.zeros_like(albedo_data, dtype=np.float32)
                                    resampled_elevation = np.zeros_like(albedo_data, dtype=np.float32)

                                    reproject(
                                        source=mask,
                                        destination=resampled_mask,
                                        src_transform=ismip_1km.rio.transform(),
                                        dst_transform=dataset.transform,
                                        src_crs=ismip_1km.rio.crs,
                                        dst_crs=dataset.crs,
                                        resampling=Resampling.nearest
                                    )

                                    reproject(
                                        source=regions,
                                        destination=resampled_regions,
                                        src_transform=regions_file.rio.transform(),
                                        dst_transform=dataset.transform,
                                        src_crs=regions_file.rio.crs,
                                        dst_crs=dataset.crs,
                                        resampling=Resampling.nearest
                                    )

                                    reproject(
                                        source=elevation,
                                        destination=resampled_elevation,
                                        src_transform=ismip_1km.rio.transform(),
                                        dst_transform=dataset.transform,
                                        src_crs=ismip_1km.rio.crs,
                                        dst_crs=dataset.crs,
                                        resampling=Resampling.nearest
                                    )
                                else:
                                    resampled_mask = mask
                                    resampled_regions = regions
                                    resampled_elevation = elevation

                                region_mask = (resampled_regions == target_region) & (resampled_mask > 0)
                                if elevation_above:
                                    elevation_mask = resampled_elevation >= 2000
                                else:
                                    elevation_mask = resampled_elevation < 2000

                                final_mask = region_mask & elevation_mask
                                filtered_albedo = np.where(final_mask, albedo_data, np.nan)

                                mean_albedo = np.nanmean(filtered_albedo)
                                if not np.isnan(mean_albedo):
                                    temp_albedos.append(mean_albedo)

            if len(temp_albedos) > 1:
                local_range = np.nanmax(temp_albedos) - np.nanmin(temp_albedos)
                max_range = max(max_range, local_range)

#loop over months, regions, and elevation thresholds
for month_num in range(4, 11):  
    target_month = f"{month_num:02d}"
    month_name = calendar.month_name[month_num]

    for elevation_above in [True, False]:
        for target_region in range(1, 9):

            years = []
            albedos = []

            for root, _, files in os.walk(base_directory):
                for filename in sorted(files):
                    if filename.endswith(".tif") and "_qa" not in filename:
                        parts = filename.split("-")
                        if len(parts) == 3 and parts[2][:2] == target_month:
                            year = int(parts[1])

                            file_path = os.path.join(root, filename)
                            with rasterio.open(file_path) as dataset:
                                albedo_data = dataset.read(1)
                                albedo_data = np.where(albedo_data == dataset.nodata, np.nan, albedo_data)

                                if mask.shape != albedo_data.shape:
                                    resampled_mask = np.zeros_like(albedo_data, dtype=np.float32)
                                    resampled_regions = np.zeros_like(albedo_data, dtype=np.float32)
                                    resampled_elevation = np.zeros_like(albedo_data, dtype=np.float32)

                                    reproject(
                                        source=mask,
                                        destination=resampled_mask,
                                        src_transform=ismip_1km.rio.transform(),
                                        dst_transform=dataset.transform,
                                        src_crs=ismip_1km.rio.crs,
                                        dst_crs=dataset.crs,
                                        resampling=Resampling.nearest
                                    )

                                    reproject(
                                        source=regions,
                                        destination=resampled_regions,
                                        src_transform=regions_file.rio.transform(),
                                        dst_transform=dataset.transform,
                                        src_crs=regions_file.rio.crs,
                                        dst_crs=dataset.crs,
                                        resampling=Resampling.nearest
                                    )

                                    reproject(
                                        source=elevation,
                                        destination=resampled_elevation,
                                        src_transform=ismip_1km.rio.transform(),
                                        dst_transform=dataset.transform,
                                        src_crs=ismip_1km.rio.crs,
                                        dst_crs=dataset.crs,
                                        resampling=Resampling.nearest
                                    )
                                else:
                                    resampled_mask = mask
                                    resampled_regions = regions
                                    resampled_elevation = elevation

                                region_mask = (resampled_regions == target_region) & (resampled_mask > 0)
                                if elevation_above:
                                    elevation_mask = resampled_elevation >= 2000
                                else:
                                    elevation_mask = resampled_elevation < 2000

                                final_mask = region_mask & elevation_mask
                                filtered_albedo = np.where(final_mask, albedo_data, np.nan)

                                mean_albedo = np.nanmean(filtered_albedo)
                                years.append(year)
                                albedos.append(mean_albedo)

            #sort and compute regression
            if len(years) > 0:
                years = np.array(years)
                albedos = np.array(albedos)
                sorted_indices = np.argsort(years)
                years = years[sorted_indices]
                albedos = albedos[sorted_indices]

                slope, intercept, r_value, p_value, std_err = linregress(years, albedos)
                trendline = slope * years + intercept

                #plot
                plt.figure(figsize=(10, 6))
                plt.rcParams['axes.titleweight'] = 'bold'
                plt.rcParams['axes.labelweight'] = 'bold'
                plt.plot(years, albedos, marker="o", linestyle="-", label=f"{region_names[target_region]}")
                plt.plot(years, trendline, linestyle="--", label=f"Trend (slope={slope:.5f}, r={r_value:.3f}, p={p_value:.3f})")
                plt.xlabel("Year")
                plt.ylabel("Mean Albedo")
                plt.title(f"Albedo Trends ({region_names[target_region]}, {month_name} 2000–2024)\nElevation {'≥' if elevation_above else '<'} 2000m")
                plt.legend()
                plt.grid(True)

                elev_str = "above" if elevation_above else "below"
                filename = f"{region_names[target_region]}_{month_name}_elev_{elev_str}_2000m.png".replace(" ", "_")
                std_dev = np.nanstd(albedos)
                mean_val = np.nanmean(albedos)
                ax = plt.gca()
                ax.fill_between(years, trendline - std_dev, trendline + std_dev, color='gray', alpha=0.3, label='±1 SD')
                ax.set_ylim(mean_val - max_range / 2, mean_val + max_range / 2)
                plt.savefig(os.path.join(output_dir, filename), dpi=300)
                plt.close()


