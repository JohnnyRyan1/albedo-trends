#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train MLP

"""

# Import modules
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

#%%

# Define user
user = 'jr555'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'

# Define files
t2m = pd.read_csv(path + 'era5/t2m.csv', index_col=['datetime'], parse_dates=True)
sf_w = pd.read_csv(path + 'era5/sf_w.csv', index_col=['datetime'], parse_dates=True)
sf_s = pd.read_csv(path + 'era5/sf_s.csv', index_col=['datetime'], parse_dates=True)
mcd = pd.read_csv(path + 'satellite/mcd43a3.csv', index_col=0, parse_dates=True)
gbi = pd.read_csv(path + 'indices/gbi-summer.csv', index_col=['Date'], parse_dates=True)
mcd = mcd[mcd.index.month.isin([6, 7, 8])]
mcd_summer = mcd.resample('YE').mean()
mcd_count = mcd.resample('YE').count()
mcd_summer[mcd_count != 3] = np.nan
mcd_summer = mcd_summer[2:]

# Mask 2002-2024
t2m_modern = t2m[t2m.index.isin(mcd_summer.index)]
sf_w_modern = sf_w[sf_w.index.isin(mcd_summer.index)]
sf_s_modern = sf_s[sf_s.index.isin(mcd_summer.index)]
gbi_modern = gbi[gbi.index.isin(mcd_summer.index)]

# Format training data
X = pd.concat([t2m_modern['region6_abl'], sf_w_modern['region6_abl'], 
               sf_s_modern['region6_abl'], gbi_modern['Value']], axis=1)
X.columns = ['t2m', 'sf_w', 'sf_s', 'gbi']

X = pd.DataFrame(gbi_modern['Value'])
y = mcd_summer['region6_abl']
y.name = 'albedo'

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# Shuffle and split: 20 train, 3 test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=3, random_state=1)

#%%

def define_optimized_mlp(input_dim):
    num_layers = 3
    base_units = 64
    residual_blocks = 0
    activation = 'leaky_relu'
    learning_rate = 0.01
    dropout_rate = 0.2

    # Build layer sizes (flat)
    layer_sizes = [base_units] * num_layers

    def get_activation_layer(act):
        if act == 'leaky_relu':
            return layers.LeakyReLU(negative_slope=0.01)
        else:
            return layers.Activation(act)

    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(layer_sizes[0], kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.LayerNormalization()(x)
    x = get_activation_layer(activation)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Residual block (only applied to first layer size)
    for _ in range(residual_blocks):
        residual = x
        x = layers.Dense(layer_sizes[0], kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.LayerNormalization()(x)
        x = get_activation_layer(activation)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Add()([x, residual])

    # Remaining dense layers
    for units in layer_sizes[1:]:
        x = layers.Dense(units, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = get_activation_layer(activation)(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

mlp_model = define_optimized_mlp(input_dim=X_train.shape[1])
history = mlp_model.fit(X_train, y_train, validation_split=0.2, epochs=1000)

#%%

def predict_with_uncertainty(f_model, x, n_iter=100):
    preds = np.stack([f_model(x, training=True).numpy().squeeze() for _ in range(n_iter)], axis=0)
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)
    return pred_mean, pred_std

mean_preds, std_preds = predict_with_uncertainty(mlp_model, X_test, n_iter=100)

for i, (mean, std) in enumerate(zip(mean_preds, std_preds)):
    print(f"Sample {i+1}: Predicted Albedo = {mean:.3f} ± {std:.3f}")
    
# Predict on test set
y_pred = mlp_model.predict(X_test)
baseline_mse = mean_squared_error(y_test, y_pred)
print(f"Baseline MLP Test MSE: {baseline_mse:.4f}")

#%%

# Create a figure and axes
fig, ax = plt.subplots()

# Plot with error bars
ax.errorbar(X_test.index.year, mean_preds, yerr=std_preds, fmt='o', label='MLP')
ax.scatter(X_test.index.year, y_test, color='red', label='True Albedo')

# Set labels, legend, and grid
ax.set_xlabel("Test year")
ax.set_ylabel("Albedo")
ax.legend()
ax.grid(True)

#%%

"""
Hindcast back to 1941.

"""

X_hindcast = pd.concat([t2m['region6_abl'][7:], sf_w['region6_abl'][7:], sf_s['region6_abl'][7:], gbi['Value']], axis=1)
X_hindcast.columns = ['t2m', 'sf_w', 'sf_s', 'gbi']
X_hindcast_scaled = pd.DataFrame(scaler.fit_transform(X_hindcast), columns=X_hindcast.columns, index=X_hindcast.index)
y_hindcast_mlp = mlp_model.predict(X_hindcast_scaled)

# Create a figure and axes
fig, ax = plt.subplots()

# Plot MLP prediction
ax.plot(X_hindcast.index, y_hindcast_mlp, label='MLP')
ax.plot(y.index, y, label='True')

# Set legend and grid
ax.legend()
ax.grid(True)

#%%

"""
Plot just 2020-2024 for comparison between truth and predicted

"""

# Create a figure and axes
fig, ax = plt.subplots()

# Plot MLP prediction
ax.plot(X_hindcast.index[54:84], y_hindcast_mlp[54:84], label='MLP')
ax.plot(y.index, y, label='True')

# Set legend and grid
ax.legend()
ax.grid(True)

#%%

"""
Plot scatter plots

"""

# Create a figure and axes
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 4), sharey=True,
                                    layout='constrained')

# Plot MLP prediction
ax1.scatter(t2m['region6_abl'], y_hindcast_mlp, label='MLP')
ax1.scatter(t2m['region6_abl'][61:84], y, label='Observed')

ax2.scatter(sf_w['region6_abl'], y_hindcast_mlp, label='MLP')
ax2.scatter(sf_w['region6_abl'][61:84], y, label='Observed')

ax3.scatter(sf_s['region6_abl'], y_hindcast_mlp, label='MLP')
ax3.scatter(sf_s['region6_abl'][61:84], y, label='Observed')

# Set legend and grid
for ax in [ax1, ax2, ax3]:
    ax.legend()
    ax.grid(True)

#%%


#%%




