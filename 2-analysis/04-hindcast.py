#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hindcast back to 1941.

"""

# Import modules
import pandas as pd
import glob
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

# Define models
mlp_models = sorted(glob.glob(path + 'data/models/*.keras'))
lin_models = sorted(glob.glob(path + 'data/models/*.joblib'))

#%%

for file in hindcast_files:
    
    # Get station name
    s = os.path.basename(file)[:-4]
    
    # Load models
    if os.path.exists(path + 'data/models/' + s + '_mlp' + '.keras'):
        
        mlp_model = tf.keras.models.load_model(path + 'data/models/' + s + '_mlp' + '.keras')
        linear_model = joblib.load(path + 'data/models/' + s + '_linear' + '.joblib')
        
        # Hindcast data
        hindcast_df = pd.read_csv(path + 'data/era5/station/' + s + '.csv',
                                  index_col=['datetime'], parse_dates=['datetime'])
        
        # Format training data
        X = pd.concat([hindcast_df['t2m'], hindcast_df['sf_winter'], 
                       hindcast_df['sf_summer']], axis=1)
        X.columns = ['t2m', 'sf_w', 'sf_s']
    
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        # Hindcast
        y_pred_lin = linear_model.predict(X_scaled)
        y_pred_mlp = mlp_model.predict(X_scaled)
        
        # Save hindcast data
        hindcast_df['y_pred_lin'] = y_pred_lin
        hindcast_df['y_pred_mlp'] = y_pred_mlp
        hindcast_df.to_csv(path + 'data/hindcast/' + s + '.csv')

        # Define colour map
        c1 = '#E05861'
        c2 = '#616E96'
        c3 = '#F8A557'
        c4 = '#3CBEDD'
    
        # Create a figure and axes
        fig, ax1 = plt.subplots(figsize=(7,4), layout='constrained')
    
        # Plot with error bars
        ax1.plot(hindcast_df.index.year, y_pred_mlp, color=c1, lw=1.5, label='MLP', marker='o', linestyle='-', alpha=0.7)
        ax1.plot(hindcast_df.index.year, y_pred_lin, color=c2, lw=1.5, label='Linear', marker='o', linestyle='-', alpha=0.7)
    
        # Set labels, legend, and grid
        ax1.set_ylabel("Albedo")
        ax1.legend()
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.grid(True, which="both", linestyle="--", linewidth=1, zorder=1)
        ax1.set_ylabel("Albedo", fontsize=12)  
    
        ax1.text(0.5, 0.98, s,
            transform=ax1.transAxes,
            fontsize=14,
            verticalalignment='top',
            horizontalalignment='right')
    
        plt.savefig(path + 'figures/3-hindcasts/' + s + '.png')
    
    else:
        pass
   
    
   
#%%










