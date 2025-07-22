#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train CAE.

"""

# Import modules
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

#%%

# Define user
user = 'johnnyryan'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'

# Define files
t2m = pd.read_csv(path + 'era5/t2m.csv', index_col=['datetime'], parse_dates=True)
sf_w = pd.read_csv(path + 'era5/sf_w.csv', index_col=['datetime'], parse_dates=True)
sf_s = pd.read_csv(path + 'era5/sf_s.csv', index_col=['datetime'], parse_dates=True)
mcd = pd.read_csv(path + 'satellite/mcd43a3.csv', index_col=0, parse_dates=True)
mcd = mcd[mcd.index.month.isin([6, 7, 8])]
mcd_summer = mcd.resample('YE').mean()
mcd_count = mcd.resample('YE').count()
mcd_summer[mcd_count != 3] = np.nan

t2m_modern = t2m[t2m.index.isin(mcd_summer.index)]
sf_w_modern = sf_w[sf_w.index.isin(mcd_summer.index)]
sf_s_modern = sf_s[sf_s.index.isin(mcd_summer.index)]

#%%

X = pd.concat([t2m_modern['region6_abl'], sf_w_modern['region6_abl'], sf_s_modern['region6_abl']], axis=1)
X.columns = ['t2m', 'sf_w', 'sf_s']
y = mcd_summer['region6_abl']
y.name = 'albedo'

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# Shuffle and split: 20 train, 5 test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=5, random_state=42)


def build_bayesian_model(input_dim, dropout_rate=0.2):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(dropout_rate),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


model = build_bayesian_model(input_dim=X_train.shape[1])
model.fit(X_train, y_train, epochs=1000)

#%%


def build_advanced_mlp(input_dim, dropout_rate=0.2):
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Residual block
    residual = x
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Add()([x, residual])  # Residual connection

    x = layers.Dense(32, activation='gelu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model


mlp_model = build_advanced_mlp(input_dim=X_train.shape[1])
mlp_model.fit(X_train, y_train, validation_split=0.2, epochs=1000)

#%%

y_pred = model.predict(X_test).squeeze()

print("R²:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))

#%%

def predict_with_uncertainty(f_model, x, n_iter=100):
    preds = np.stack([f_model(x, training=True).numpy().squeeze() for _ in range(n_iter)], axis=0)
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)
    return pred_mean, pred_std

mean_preds, std_preds = predict_with_uncertainty(mlp_model, X_test, n_iter=100)

for i, (mean, std) in enumerate(zip(mean_preds, std_preds)):
    print(f"Sample {i+1}: Predicted Albedo = {mean:.3f} ± {std:.3f}")
    


#%%

rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train, y_train)


y_pred_rf = rf.predict(X_test)

print("Random Forest R²:", r2_score(y_test, y_pred_rf))
print("Random Forest RMSE:", mean_squared_error(y_test, y_pred_rf))

# Get predictions from each tree
tree_preds = np.stack([tree.predict(X_test) for tree in rf.estimators_])
rf_mean = tree_preds.mean(axis=0)
rf_std = tree_preds.std(axis=0)

# Example output
for i in range(len(y_test)):
    print(f"Predicted: {rf_mean[i]:.3f} ± {rf_std[i]:.3f} | True: {y_test.iloc[i]:.3f}")

#%%

plt.errorbar(range(len(y_test)), mean_preds, yerr=std_preds, fmt='o', label='MLP')
plt.errorbar(range(len(y_test)), rf_mean, yerr=rf_std, fmt='o', label='RF')

plt.scatter(range(len(y_test)), y_test, color='red', label='True Albedo')

plt.xlabel("Test Sample")
plt.ylabel("Albedo")
plt.legend()
plt.grid(True)
plt.show()

#%%

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4, 5, None],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}


rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    rf, param_grid, 
    cv=5,
    scoring='neg_root_mean_squared_error',  # or 'r2'
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best RMSE (CV):", -grid_search.best_score_)  # negate because we used "neg RMSE"


best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("Test R²:", r2_score(y_test, y_pred_rf))
print("Test RMSE:", mean_squared_error(y_test, y_pred_rf))


#%%

X_hindcast = pd.concat([t2m['region6_abl'], sf_w['region6_abl'], sf_s['region6_abl']], axis=1)
X_hindcast.columns = ['t2m', 'sf_w', 'sf_s']
X_hindcast_scaled = pd.DataFrame(scaler.fit_transform(X_hindcast), columns=X_hindcast.columns, index=X_hindcast.index)

y_hindcast = best_rf.predict(X_hindcast_scaled)
y_hindcast_mlp = mlp_model.predict(X_hindcast_scaled)

#%%

plt.plot(X_hindcast.index, y_hindcast, label='RF')
plt.plot(X_hindcast.index, y_hindcast_mlp, label='MLP')
plt.plot(y.index, y, label='True')

plt.legend()
plt.grid(True)
plt.show()

#%%

plt.plot(X_hindcast.index[59:84], y_hindcast[59:84], label='RF')
plt.plot(X_hindcast.index[59:84], y_hindcast_mlp[59:84], label='MLP')
plt.plot(y.index, y, label='True')

plt.legend()
plt.grid(True)
plt.show()


