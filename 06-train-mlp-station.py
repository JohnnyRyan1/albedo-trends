#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train MLP

"""

# Import modules
import pandas as pd
import numpy as np
import glob
import os
import tensorflow as tf
from tensorflow.keras import layers, regularizers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import joblib

#%%

# Define user
user = 'johnnyryan'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/'

# Define files
files = sorted(glob.glob(path + 'data/station/*.csv'))

# Define files
hindcast_files = sorted(glob.glob(path + 'data/era5/station/*.csv'))

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

def predict_with_uncertainty(f_model, x, n_iter=100):
    preds = np.stack([f_model(x, training=True).numpy().squeeze() for _ in range(n_iter)], axis=0)
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)
    return pred_mean, pred_std


#%%

rmse_mlp = []
rmse_linear = []
mean_albedo = []
station = []

for file in files:
    
    # Get station name
    s = os.path.basename(file)[:-4]
    
    # Read file
    df = pd.read_csv(file, index_col=['datetime'], parse_dates=['datetime'])
    df = df[2:]
    
    # Compute differences between AWS and MCD
    mask = np.isfinite(df['aws']) & np.isfinite(df['mcd'])
    if np.sum(np.isfinite(df['aws'])) > 5:
        
        print('Processing... %s' % s)
        
        # Append station
        station.append(s)
        mean_albedo.append(df['mcd'].mean())
        
        # Format training data
        X = pd.concat([df['t2m'], df['sf_winter'], 
                       df['sf_summer']], axis=1)
        X.columns = ['t2m', 'sf_w', 'sf_s']
    
        y = df['mcd']
        y.name = 'albedo'
    
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
        # Shuffle and split: 20 train, 3 test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=3, random_state=1)
        
        # Fit MLP
        mlp_model = define_optimized_mlp(input_dim=X_train.shape[1])
        mlp_model.fit(X_train, y_train, validation_split=0.2, epochs=1000, verbose=0)
            
        # Predict on test set
        y_pred = mlp_model.predict(X_test)
        rmse_mlp.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        
        # Fit linear model
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        
        # Save model
        mlp_model.save(path + 'data/models/' + s + '_mlp' + '.keras')
        joblib.dump(linear_model, path + 'data/models/' + s + '_linear' + '.joblib')
        
        # Predict and calculate residuals
        y_pred_linear = linear_model.predict(X_test)

        # Calculate RMSE
        rmse_linear.append(np.sqrt(mean_squared_error(y_test, y_pred_linear)))
        
        # Define colour map
        c1 = '#E05861'
        c2 = '#616E96'
        c3 = '#F8A557'
        c4 = '#3CBEDD'

        # Create a figure and axes
        fig, ax1 = plt.subplots(figsize=(7,4), layout='constrained')

        # Plot with error bars
        ax1.plot(df.index.year, df['mcd'], color='black', lw=1.5, marker='o', linestyle='-', alpha=0.7, label='True albedo')
        ax1.plot(df.index.year, mlp_model.predict(X_scaled), color=c1, lw=1.5, label='MLP', marker='o', linestyle='-', alpha=0.7)
        ax1.plot(df.index.year, linear_model.predict(X_scaled), color=c2, lw=1.5, label='Linear', marker='o', linestyle='-', alpha=0.7)

        # Set labels, legend, and grid
        ax1.set_ylabel("Albedo")
        ax1.legend()
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.grid(True, which="both", linestyle="--", linewidth=1, zorder=1)
        ax1.set_ylabel("Albedo", fontsize=12)  
        ax1.set_xlim(2001,2025)

        ax1.text(0.5, 0.98, s,
            transform=ax1.transAxes,
            fontsize=14,
            verticalalignment='top',
            horizontalalignment='right')

        plt.savefig(path + 'figures/2-models/' + s + '.png')


#%%

# Make DataFrame
df = pd.DataFrame(list(zip(station, rmse_linear, rmse_mlp, mean_albedo)), 
                      columns=['station', 'linear', 'mlp', 'mean_albedo'])

# Save
df.to_csv(path + 'data/deep-learning/model-rmse.csv')

#%%

mean_preds, std_preds = predict_with_uncertainty(mlp_model, X_test, n_iter=100)

for i, (mean, std) in enumerate(zip(mean_preds, std_preds)):
    print(f"Sample {i+1}: Predicted Albedo = {mean:.3f} ± {std:.3f}")
            
# Create a figure and axes
fig, ax = plt.subplots(figsize=(4,4))

# Plot with error bars
ax.errorbar(np.arange(0,3,1), mean_preds, yerr=std_preds, fmt='o', label='MLP')
ax.scatter(np.arange(0,3,1), y_test, color='red', label='True Albedo')
ax.scatter(np.arange(0,3,1), y_pred_linear, color='green', label='Linear')


# Set labels, legend, and grid
ax.set_xlabel("Test year")
ax.set_ylabel("Albedo")
ax.legend()
ax.grid(True)


#%%




