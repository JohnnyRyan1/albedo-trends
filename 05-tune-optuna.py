#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Hyperparameter tuning using Optuna

"""

# Import modules
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import optuna

# Define user
user = 'jr555'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'

# Define functions
def build_mlp_model(
    input_dim,
    layer_sizes,
    num_residual_blocks,
    activation,
    dropout_rate,
    learning_rate
):
    inputs = tf.keras.Input(shape=(input_dim,))
    
    def get_activation_layer(act):
        if act == 'leaky_relu':
            return layers.LeakyReLU(negative_slope=0.01)
        else:
            return layers.Activation(act)

    x = layers.Dense(layer_sizes[0], kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.LayerNormalization()(x)
    x = get_activation_layer(activation)(x)
    x = layers.Dropout(dropout_rate)(x)

    for _ in range(num_residual_blocks):
        residual = x
        x = layers.Dense(layer_sizes[0], kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.LayerNormalization()(x)
        x = get_activation_layer(activation)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Add()([x, residual])

    for units in layer_sizes[1:]:
        x = layers.Dense(units, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = get_activation_layer(activation)(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

def benchmark_model(
    X_train, y_train, X_test, y_test,
    layer_sizes, num_residual_blocks,
    activation, dropout_rate, learning_rate
):
    model_name = (
        f"Layers: {layer_sizes}, Res: {num_residual_blocks}, Act: {activation}, "
        f"Drop: {dropout_rate}, LR: {learning_rate}"
    )
    print(f"\n🔧 Testing → {model_name}")

    model = build_mlp_model(
        input_dim=X_train.shape[1],
        layer_sizes=layer_sizes,
        num_residual_blocks=num_residual_blocks,
        activation=activation,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=1000,
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, y_pred)
    print(f"📉 {model_name} → Test MSE: {mse:.4f}")
    return mse

def generate_layer_sizes(num_layers, base_units, taper):
    return [base_units // (2 ** i) for i in range(num_layers)] if taper else [base_units] * num_layers

def objective(trial):
    # Sample hyperparameters
    num_layers = trial.suggest_int('num_layers', 2, 5)
    base_units = trial.suggest_categorical('base_units', [32, 64, 128])
    taper = trial.suggest_categorical('taper', [True, False])
    residual_blocks = trial.suggest_int('residual_blocks', 0, 1)
    activation = trial.suggest_categorical('activation', ['gelu', 'leaky_relu'])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-4, 1e-3, 1e-2])
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.2, 0.3])

    layer_sizes = generate_layer_sizes(num_layers, base_units, taper)

    # Evaluate model
    mse = benchmark_model(
        X_train, y_train, X_test, y_test,
        layer_sizes=layer_sizes,
        num_residual_blocks=residual_blocks,
        activation=activation,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    return mse

# Define files
t2m = pd.read_csv(path + 'era5/t2m.csv', index_col=['datetime'], parse_dates=True)
sf_w = pd.read_csv(path + 'era5/sf_w.csv', index_col=['datetime'], parse_dates=True)
sf_s = pd.read_csv(path + 'era5/sf_s.csv', index_col=['datetime'], parse_dates=True)
mcd = pd.read_csv(path + 'satellite/mcd43a3.csv', index_col=0, parse_dates=True)
mcd = mcd[mcd.index.month.isin([6, 7, 8])]
mcd_summer = mcd.resample('YE').mean()
mcd_count = mcd.resample('YE').count()
mcd_summer[mcd_count != 3] = np.nan
mcd_summer = mcd_summer[2:]

# Mask 2002-2024
t2m_modern = t2m[t2m.index.isin(mcd_summer.index)]
sf_w_modern = sf_w[sf_w.index.isin(mcd_summer.index)]
sf_s_modern = sf_s[sf_s.index.isin(mcd_summer.index)]

# Format training data
X = pd.concat([t2m_modern['region6_abl'], sf_w_modern['region6_abl'], sf_s_modern['region6_abl']], axis=1)
X.columns = ['t2m', 'sf_w', 'sf_s']
y = mcd_summer['region6_abl']
y.name = 'albedo'

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# Shuffle and split: 20 train, 3 test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=3, random_state=1)

study = optuna.create_study(direction='minimize')
study.optimize(objective, timeout=7200)  # 2 hours
#study.optimize(objective, n_trials=50)

# Print best result
print("✅ Best trial:")
print(study.best_trial.params)
print(f"Lowest Test MSE: {study.best_value:.4f}")

# Save
best_params_df = pd.DataFrame([study.best_trial.params])
best_params_df.to_csv(path + 'params.csv')

all_trials = []

for trial in study.trials:
    if trial.state.name == "COMPLETE":
        row = trial.params.copy()
        row['Test MSE'] = trial.value
        all_trials.append(row)

all_trials_df = pd.DataFrame(all_trials)
print("📊 All Trials:")
print(all_trials_df.sort_values(by="Test MSE"))
all_trials_df.to_csv(path + 'deep-learning/tuning1.csv')
                     
                     
                     
                     
                     
                     