#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train CAE.

"""

# Import modules
import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

#%%

# Define user
user = 'jr555'

# Define path
path1 = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'
path2 = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/hydrology/data/'
path3 = '/Volumes/meltwater-mapping_satellite-data/data/'

# Import ISMIP 1 km grid
ismip_1km = xr.open_dataset(path2 + '1km-ISMIP6-GIMP.nc')

# Define mask
mask = ismip_1km['GIMP'].values

# Define ERA5 data
era_t2m = xr.open_dataset(path1 + 'era5/era-summer-t2m-1941-2025.nc')
era_summer_sf = xr.open_dataset(path1 + 'era5/era-summer-sf-1941-2025.nc')
era_winter_sf = xr.open_dataset(path1 + 'era5/era-winter-sf-1941-2025.nc')

# Define MCD data
mcd = xr.open_dataset(path3 + 'MCD43A3/mosaics/summer/mcd43a3-summer-albedo.nc')

# Define years
years = np.arange(2000, 2025, 1)

# Define config
epochs = 20
optimizer = 'adam'
loss = 'mse'
img_size = (320, 320)
img_height = 320
img_width = 320
img_bands = 3

#%%

"""
Tile

"""

def sliding_windows(arr, block_size=320, overlap=0.1):
    """
    Split 2D array into overlapping tiles.

    Parameters:
        arr (np.ndarray): 2D input array.
        block_size (int): size of square block (default 320).
        overlap (float): fraction of overlap (default 0.1 = 10%).

    Returns:
        List of tuples: (block_array, (y_start, x_start))
    """
    step = int(block_size * (1 - overlap))
    y_max, x_max = arr.shape
    tiles = []

    y_starts = list(range(0, y_max - block_size + 1, step))
    x_starts = list(range(0, x_max - block_size + 1, step))

    # Make sure last block covers the end if not exactly divisible
    if y_starts[-1] + block_size < y_max:
        y_starts.append(y_max - block_size)
    if x_starts[-1] + block_size < x_max:
        x_starts.append(x_max - block_size)

    for y in y_starts:
        for x in x_starts:
            block = arr[y:y+block_size, x:x+block_size]
            tiles.append((block, (y, x)))

    return tiles


arr = np.random.rand(2881, 1681)
tiles = sliding_windows(mask, block_size=320, overlap=0.1)

print(f"Extracted {len(tiles)} tiles")


#%%

"""

For each tile:
    
    1. Make a list of arrays containing:
        Features: winter snowfall, summer snowfall, summer air temperature
        Input image: summer albedo

    2. Scale features between 0 and 1
    
    4. Split and shuffle into training, testing, and validation

    5. Skip tile if mask is all False

"""
t = 2

tile_y = tiles[t][1][0]
tile_x = tiles[t][1][1]

# Mask albedo
albedo_tile = mcd['albedo'][:,tile_y:tile_y+320, tile_x:tile_x+320]
era_summer_sf_tile = era_summer_sf['sf'][59:84,tile_y:tile_y+320, tile_x:tile_x+320]
era_winter_sf_tile = era_winter_sf['sf'][59:84,tile_y:tile_y+320, tile_x:tile_x+320]
era_summer_t2m_tile = era_t2m['t2m'][59:84,tile_y:tile_y+320, tile_x:tile_x+320]

X = np.stack([era_summer_t2m_tile.values, era_summer_sf_tile.values, era_winter_sf_tile.values], axis=-1)
y = albedo_tile.values

X_norm = np.empty_like(X)
for c in range(X.shape[-1]):
    band = X[..., c]
    band_min = np.nanmin(band)
    band_max = np.nanmax(band)
    X_norm[..., c] = (band - band_min) / (band_max - band_min)

# Shuffle the indices
np.random.seed(42)
indices = np.random.permutation(len(X))

# Define split sizes
train_idx = indices[:20]
val_idx = indices[20:22]
test_idx = indices[22:]

# Slice the arrays
X_train, y_train = X_norm[train_idx], y[train_idx]
X_val, y_val     = X_norm[val_idx], y[val_idx]
X_test, y_test   = X_norm[test_idx], y[test_idx]

# Convert to tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(100).batch(4).prefetch(tf.data.AUTOTUNE)
val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(2).prefetch(tf.data.AUTOTUNE)
test_ds  = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(3).prefetch(tf.data.AUTOTUNE)

print('Test years for this tile are %s' % str(years[test_idx]))


#%%

def masked_mse(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return tf.reduce_mean(tf.square(y_true - y_pred))

def unet_autoencoder(input_shape=(img_width, img_height, img_bands)):
    inputs = layers.Input(shape=input_shape)

    # ---- Encoder ----
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)

    # ---- Bottleneck ----
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(b)

    # ---- Decoder ----
    u3 = layers.UpSampling2D()(b)
    u3 = layers.Concatenate()([u3, c3])
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(u3)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(c5)

    u1 = layers.UpSampling2D()(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(c6)

    # ---- Output ----
    outputs = layers.Conv2D(1, 1, activation='sigmoid', name='albedo')(c6)

    return models.Model(inputs, outputs)

model = unet_autoencoder()
model.compile(optimizer=optimizer, loss=masked_mse, metrics=['mse'])

# Train
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Evaluate
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ Final test accuracy: {test_acc:.4f}")

#%%

# Make prediction
prediction = model.predict(test_ds)

#%%

"""

For each tile:
    
    1. Define a CAE based on U-Net architecture and sigmoid activation function

    2. Initialize model with random weights using He initialisation
    
    3. MSE loss function

He, K., Zhang, X., Ren, S. & Sun, J. Delving Deep into Rectifiers: Surpassing 
Human-Level Performance on ImageNet Classification. in IEEE International 
Conference on Computer Vision (ICCV) https://doi.org/10.1109/ICCV.2015.123 1026–1034 (2015).

"""



#%%


"""

For each model:
    
    1. Predict albedo for 1941-2024 period, noting which years were not used for training
    
"""


#%%


"""

For each tile:
    
    1. Compute ensemble mean and std dev for 1941-2024 period 
    
    2. Evaluate ensemble mean against years that were not used for training

"""

#%%

"""

For each tile:
    
    1. Merge the ensemble means for all tiles + std dev (averaging overlapping regions) 
    
    2. Compute statistics

"""



#%%





