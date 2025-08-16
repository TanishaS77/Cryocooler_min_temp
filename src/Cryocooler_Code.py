#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load your cleaned dataset
df = pd.read_csv("master_cryocooler_data.csv")

# Feature Engineering: Add lagged feature and refined transition detection
df['prev_min_temp'] = df['min_temp'].shift(1)
df['temp_change'] = df['min_temp'].diff().abs()
df['is_transition'] = (df['temp_change'] > 20).astype(int)  # Threshold can be adjusted
df = df.dropna()  # Drop rows with NaN from lagged feature

# Define features and target
X = df[['time', 'power_input', 'chx_temp', 'outlet_temp', 'pr_regen', 'prev_min_temp', 'is_transition']]
y = df['min_temp']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Model 2: Optimized Random Forest with GridSearchCV and KFold
param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}  # Reduced complexity
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=KFold(n_splits=10, shuffle=True, random_state=42), scoring='r2')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("Best params:", grid_search.best_params_)
print("Linear MAE:", mean_absolute_error(y_test, y_pred_lr))
print("Linear R2:", r2_score(y_test, y_pred_lr))
print("RF MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RF R2:", r2_score(y_test, y_pred_rf))

# Cross-validation with KFold
cv_scores = cross_val_score(best_rf, X, y, cv=KFold(n_splits=10, shuffle=True, random_state=42), scoring='r2')
print("Cross-validated R2 (mean):", cv_scores.mean())
print("Cross-validated R2 (std):", cv_scores.std())

# Handle spikes by segmenting data (by power_input)
segments = df.groupby('power_input')
for power, segment_df in segments:
    segment_X = segment_df[['time', 'power_input', 'chx_temp', 'outlet_temp', 'pr_regen', 'prev_min_temp', 'is_transition']]
    segment_y = segment_df['min_temp']
    segment_X_train, segment_X_test, segment_y_train, segment_y_test = train_test_split(segment_X, segment_y, test_size=0.2, random_state=42)
    segment_rf = RandomForestRegressor(**grid_search.best_params_, random_state=42)
    segment_rf.fit(segment_X_train, segment_y_train)
    segment_y_pred = segment_rf.predict(segment_X_test)
    print(f"Power {power}W - RF MAE: {mean_absolute_error(segment_y_test, segment_y_pred)}")
    print(f"Power {power}W - RF R2: {r2_score(segment_y_test, segment_y_pred)}")

# Visualization: Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual', color='black', linewidth=2)
plt.plot(y_pred_rf, label='Predicted (Random Forest)', linestyle='--', color='blue')
plt.title("Predicted vs Actual Min Temp (All Data)")
plt.xlabel("Sample Index")
plt.ylabel("Min Temp (K)")
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance
importances = best_rf.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 5))
plt.bar(feature_names, importances)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()

# Residual Plot
residuals = y_test - y_pred_rf
plt.figure(figsize=(10, 5))
plt.scatter(range(len(residuals)), residuals, color='red', s=10)
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residuals of Random Forest Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.show()


# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -------------------------
# Load and Preprocess
# -------------------------
df = pd.read_csv("master_cryocooler_data.csv")

# Rolling features instead of direct prev_min_temp
df['temp_rolling_mean'] = df['min_temp'].rolling(window=5).mean()
df['temp_rolling_diff'] = df['min_temp'].diff(periods=5)
df['temp_change'] = df['min_temp'].diff().abs()
df['is_transition'] = (df['temp_change'] > 20).astype(int)

df = df.dropna()

# Define features and target
X = df[['time', 'power_input', 'chx_temp', 'outlet_temp', 'pr_regen',
        'temp_rolling_mean', 'temp_rolling_diff', 'is_transition']]
y = df['min_temp']

# -------------------------
# Time-Series Split
# -------------------------
tscv = TimeSeriesSplit(n_splits=5)

# -------------------------
# Linear Regression (Baseline)
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LinearRegression()
lr_scores = cross_val_score(lr, X_scaled, y, cv=tscv, scoring='r2')
lr.fit(X_scaled, y)
print("Linear Regression CV R²:", lr_scores.mean(), "+/-", lr_scores.std())

# -------------------------
# Random Forest with Regularization
# -------------------------
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, None],
    'min_samples_leaf': [5, 10, 20],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=tscv,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X, y)
best_rf = grid_search.best_estimator_
rf_scores = cross_val_score(best_rf, X, y, cv=tscv, scoring='r2')

print("Best RF params:", grid_search.best_params_)
print("Random Forest CV R²:", rf_scores.mean(), "+/-", rf_scores.std())

# -------------------------
# Train-Test Split (Final Evaluation)
# -------------------------
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)

print("RF Test MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RF Test R²:", r2_score(y_test, y_pred_rf))

# -------------------------
# Visualization: Actual vs Predicted
# -------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual', color='black', linewidth=2)
plt.plot(y_pred_rf, label='Predicted (Random Forest)', linestyle='--', color='blue')
plt.title("Predicted vs Actual Min Temp (Time-Series Split)")
plt.xlabel("Sample Index")
plt.ylabel("Min Temp (K)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Feature Importance
# -------------------------
importances = best_rf.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 5))
plt.bar(feature_names, importances)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()

# -------------------------
# Residual Plot
# -------------------------
residuals = y_test - y_pred_rf
plt.figure(figsize=(10, 5))
plt.scatter(range(len(residuals)), residuals, color='red', s=10)
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residuals of Random Forest Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.show()


# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load and Preprocess
df = pd.read_csv("master_cryocooler_data.csv")

# Convert 'time' to datetime if not already, and extract features
df['time'] = pd.to_datetime(df['time'])
df['hour'] = df['time'].dt.hour
df['day'] = df['time'].dt.day

# Rolling features with adjusted window
df['temp_rolling_mean'] = df['min_temp'].rolling(window=3).mean()  # Reduced to 3
df['temp_rolling_diff'] = df['min_temp'].diff(periods=3)  # Adjusted to match window
df['temp_change'] = df['min_temp'].diff().abs()
df['is_transition'] = (df['temp_change'] > 10).astype(int)  # Lowered threshold

df = df.dropna()

# Define features and target
X = df[['hour', 'day', 'power_input', 'chx_temp', 'outlet_temp', 'pr_regen',
        'temp_rolling_mean', 'temp_rolling_diff', 'is_transition']]
y = df['min_temp']

# Time-Series Split
tscv = TimeSeriesSplit(n_splits=10)  # Increased splits

# Linear Regression (Baseline)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LinearRegression()
lr_scores = cross_val_score(lr, X_scaled, y, cv=tscv, scoring='r2')
lr.fit(X_scaled, y)
print("Linear Regression CV R²:", lr_scores.mean(), "+/-", lr_scores.std())

# Random Forest with Regularization
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [7, 10],  # Moderate complexity
    'min_samples_leaf': [2, 5],  # Relaxed constraint
    'max_features': ['sqrt']
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=tscv,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X, y)
best_rf = grid_search.best_estimator_
rf_scores = cross_val_score(best_rf, X, y, cv=tscv, scoring='r2')

print("Best RF params:", grid_search.best_params_)
print("Random Forest CV R²:", rf_scores.mean(), "+/-", rf_scores.std())

# Train-Test Split (Final Evaluation)
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)

print("RF Test MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RF Test R²:", r2_score(y_test, y_pred_rf))

# Visualization: Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual', color='black', linewidth=2)
plt.plot(y_pred_rf, label='Predicted (Random Forest)', linestyle='--', color='blue')
plt.title("Predicted vs Actual Min Temp (Time-Series Split)")
plt.xlabel("Sample Index")
plt.ylabel("Min Temp (K)")
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance
importances = best_rf.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 5))
plt.bar(feature_names, importances)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()

# Residual Plot
residuals = y_test - y_pred_rf
plt.figure(figsize=(10, 5))
plt.scatter(range(len(residuals)), residuals, color='red', s=10)
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residuals of Random Forest Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.show()


# In[6]:


import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load and Preprocess
df = pd.read_csv("master_cryocooler_data.csv")

# Convert 'time' to datetime and extract cyclic features
df['time'] = pd.to_datetime(df['time'])
df['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
df['day_sin'] = np.sin(2 * np.pi * df['time'].dt.day / 31)
df['day_cos'] = np.cos(2 * np.pi * df['time'].dt.day / 31)

# Rolling features with adjusted window
df['temp_rolling_mean'] = df['min_temp'].rolling(window=5).mean()
df['temp_rolling_diff'] = df['min_temp'].diff(periods=5)  # Match window
df = df.dropna()

# Define features and target (remove is_transition for now)
X = df[['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'power_input', 'chx_temp',
        'outlet_temp', 'pr_regen', 'temp_rolling_mean', 'temp_rolling_diff']]
y = df['min_temp']

# Time-Series Split
tscv = TimeSeriesSplit(n_splits=5)  # Reduced splits

# Linear Regression (Baseline)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LinearRegression()
lr_scores = cross_val_score(lr, X_scaled, y, cv=tscv, scoring='r2')
lr.fit(X_scaled, y)
print("Linear Regression CV R²:", lr_scores.mean(), "+/-", lr_scores.std())

# Random Forest with Regularization
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [7, 10],
    'min_samples_leaf': [2, 5],
    'max_features': ['sqrt']
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=tscv,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X, y)
best_rf = grid_search.best_estimator_
rf_scores = cross_val_score(best_rf, X, y, cv=tscv, scoring='r2')

print("Best RF params:", grid_search.best_params_)
print("Random Forest CV R²:", rf_scores.mean(), "+/-", rf_scores.std())

# Train-Test Split (Final Evaluation)
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)

print("RF Test MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RF Test R²:", r2_score(y_test, y_pred_rf))

# Visualization: Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual', color='black', linewidth=2)
plt.plot(y_pred_rf, label='Predicted (Random Forest)', linestyle='--', color='blue')
plt.title("Predicted vs Actual Min Temp (Time-Series Split)")
plt.xlabel("Sample Index")
plt.ylabel("Min Temp (K)")
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance
importances = best_rf.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 5))
plt.bar(feature_names, importances)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()

# Residual Plot
residuals = y_test - y_pred_rf
plt.figure(figsize=(10, 5))
plt.scatter(range(len(residuals)), residuals, color='red', s=10)
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residuals of Random Forest Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.show()


# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load your cleaned dataset
df = pd.read_csv("master_cryocooler_data.csv")

# Feature Engineering: Add lagged feature and refined transition detection
df['prev_min_temp'] = df['min_temp'].shift(1)
df['temp_change'] = df['min_temp'].diff().abs()
df['is_transition'] = (df['temp_change'] > 10).astype(int)  # Threshold from recent run
df = df.dropna()  # Drop rows with NaN from lagged feature

# Convert 'time' to datetime and then to numeric (seconds since epoch)
df['time'] = pd.to_datetime(df['time'], errors='coerce')  # Handle invalid parsing with NaT
df['time_numeric'] = (df['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
df = df.dropna()  # Drop any rows where 'time' conversion failed

# Define features and target
X = df[['time_numeric', 'power_input', 'chx_temp', 'outlet_temp', 'pr_regen', 'prev_min_temp', 'is_transition']]
y = df['min_temp']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Model 2: Optimized Random Forest with GridSearchCV and KFold
param_grid = {'n_estimators': [100, 150], 'max_depth': [5, 7]}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("Best params:", grid_search.best_params_)
print("Linear MAE:", mean_absolute_error(y_test, y_pred_lr))
print("Linear R2:", r2_score(y_test, y_pred_lr))
print("RF MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RF R2:", r2_score(y_test, y_pred_rf))

# Cross-validation with KFold
cv_scores = cross_val_score(best_rf, X, y, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
print("Cross-validated R2 (mean):", cv_scores.mean())
print("Cross-validated R2 (std):", cv_scores.std())

# Handle spikes by segmenting data (by power_input)
segments = df.groupby('power_input')
for power, segment_df in segments:
    segment_X = segment_df[['time_numeric', 'power_input', 'chx_temp', 'outlet_temp', 'pr_regen', 'prev_min_temp', 'is_transition']]
    segment_y = segment_df['min_temp']
    segment_X_train, segment_X_test, segment_y_train, segment_y_test = train_test_split(segment_X, segment_y, test_size=0.2, random_state=42)
    segment_rf = RandomForestRegressor(**grid_search.best_params_, random_state=42)
    segment_rf.fit(segment_X_train, segment_y_train)
    segment_y_pred = segment_rf.predict(segment_X_test)
    print(f"Power {power}W - RF MAE: {mean_absolute_error(segment_y_test, segment_y_pred)}")
    print(f"Power {power}W - RF R2: {r2_score(segment_y_test, segment_y_pred)}")

# Visualization: Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual', color='black', linewidth=2)
plt.plot(y_pred_rf, label='Predicted (Random Forest)', linestyle='--', color='blue')
plt.title("Predicted vs Actual Min Temp (All Data)")
plt.xlabel("Sample Index")
plt.ylabel("Min Temp (K)")
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance
importances = best_rf.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 5))
plt.bar(feature_names, importances)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()

# Residual Plot
residuals = y_test - y_pred_rf
plt.figure(figsize=(10, 5))
plt.scatter(range(len(residuals)), residuals, color='red', s=10)
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residuals of Random Forest Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.show()


# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load your cleaned dataset
df = pd.read_csv("master_cryocooler_data.csv")

# Feature Engineering: Add lagged feature and refined transition detection
df['prev_min_temp'] = df['min_temp'].shift(1)
df['temp_change'] = df['min_temp'].diff().abs()
df['is_transition'] = (df['temp_change'] > 10).astype(int)
df = df.dropna()  # Drop rows with NaN from lagged feature

# Convert 'time' to datetime and then to numeric (seconds since epoch)
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df['time_numeric'] = (df['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
df = df.dropna()  # Drop any rows where 'time' conversion failed

# Define features and target
X = df[['time_numeric', 'power_input', 'chx_temp', 'outlet_temp', 'pr_regen', 'prev_min_temp', 'is_transition']]
y = df['min_temp']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Model 2: Optimized Random Forest with GridSearchCV and KFold
param_grid = {'n_estimators': [150], 'max_depth': [6, 7]}  # Reduced max_depth range
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("Best params:", grid_search.best_params_)
print("Linear MAE:", mean_absolute_error(y_test, y_pred_lr))
print("Linear R2:", r2_score(y_test, y_pred_lr))
print("RF MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RF R2:", r2_score(y_test, y_pred_rf))

# Cross-validation with KFold
cv_scores = cross_val_score(best_rf, X, y, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
print("Cross-validated R2 (mean):", cv_scores.mean())
print("Cross-validated R2 (std):", cv_scores.std())

# Handle spikes by segmenting data (by power_input)
segments = df.groupby('power_input')
for power, segment_df in segments:
    segment_X = segment_df[['time_numeric', 'power_input', 'chx_temp', 'outlet_temp', 'pr_regen', 'prev_min_temp', 'is_transition']]
    segment_y = segment_df['min_temp']
    segment_X_train, segment_X_test, segment_y_train, segment_y_test = train_test_split(segment_X, segment_y, test_size=0.2, random_state=42)
    segment_rf = RandomForestRegressor(**grid_search.best_params_, random_state=42)
    segment_rf.fit(segment_X_train, segment_y_train)
    segment_y_pred = segment_rf.predict(segment_X_test)
    print(f"Power {power}W - RF MAE: {mean_absolute_error(segment_y_test, segment_y_pred)}")
    print(f"Power {power}W - RF R2: {r2_score(segment_y_test, segment_y_pred)}")

# Visualization: Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual', color='black', linewidth=2)
plt.plot(y_pred_rf, label='Predicted (Random Forest)', linestyle='--', color='blue')
plt.title("Predicted vs Actual Min Temp (All Data)")
plt.xlabel("Sample Index")
plt.ylabel("Min Temp (K)")
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance
importances = best_rf.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 5))
plt.bar(feature_names, importances)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()

# Residual Plot
residuals = y_test - y_pred_rf
plt.figure(figsize=(10, 5))
plt.scatter(range(len(residuals)), residuals, color='red', s=10)
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residuals of Random Forest Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.show()


# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Load your cleaned dataset
df = pd.read_csv("master_cryocooler_data.csv")

# Feature Engineering: Add lagged feature and refined transition detection
df['prev_min_temp'] = df['min_temp'].shift(1)
df['temp_change'] = df['min_temp'].diff().abs()
df['is_transition'] = (df['temp_change'] > 10).astype(int)  # Threshold from recent run
df = df.dropna()  # Drop rows with NaN from lagged feature

# Convert 'time' to datetime and then to numeric (seconds since epoch)
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df['time_numeric'] = (df['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
df = df.dropna()  # Drop any rows where 'time' conversion failed

# Define features and target
X = df[['time_numeric', 'power_input', 'chx_temp', 'outlet_temp', 'pr_regen', 'prev_min_temp', 'is_transition']]
y = df['min_temp']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for Linear Regression and XGBoost (if needed)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Model 2: Optimized Random Forest with GridSearchCV and KFold
rf_param_grid = {'n_estimators': [150], 'max_depth': [6, 7]}
rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
rf_grid_search.fit(X_train, y_train)
best_rf = rf_grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Model 3: Optimized XGBoost with GridSearchCV and KFold
xgb_param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}
xgb_grid_search = GridSearchCV(xgb.XGBRegressor(random_state=42, objective='reg:squarederror'), xgb_param_grid, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
xgb_grid_search.fit(X_train, y_train)
best_xgb = xgb_grid_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)

print("Best RF params:", rf_grid_search.best_params_)
print("Best XGBoost params:", xgb_grid_search.best_params_)
print("Linear MAE:", mean_absolute_error(y_test, y_pred_lr))
print("Linear R2:", r2_score(y_test, y_pred_lr))
print("RF MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RF R2:", r2_score(y_test, y_pred_rf))
print("XGBoost MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("XGBoost R2:", r2_score(y_test, y_pred_xgb))

# Cross-validation with KFold for both RF and XGBoost
rf_cv_scores = cross_val_score(best_rf, X, y, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
xgb_cv_scores = cross_val_score(best_xgb, X, y, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
print("RF Cross-validated R2 (mean):", rf_cv_scores.mean())
print("RF Cross-validated R2 (std):", rf_cv_scores.std())
print("XGBoost Cross-validated R2 (mean):", xgb_cv_scores.mean())
print("XGBoost Cross-validated R2 (std):", xgb_cv_scores.std())

# Handle spikes by segmenting data (by power_input) for XGBoost (RF already done in previous)
segments = df.groupby('power_input')
for power, segment_df in segments:
    segment_X = segment_df[['time_numeric', 'power_input', 'chx_temp', 'outlet_temp', 'pr_regen', 'prev_min_temp', 'is_transition']]
    segment_y = segment_df['min_temp']
    segment_X_train, segment_X_test, segment_y_train, segment_y_test = train_test_split(segment_X, segment_y, test_size=0.2, random_state=42)
    segment_xgb = xgb.XGBRegressor(**xgb_grid_search.best_params_, random_state=42, objective='reg:squarederror')
    segment_xgb.fit(segment_X_train, segment_y_train)
    segment_y_pred = segment_xgb.predict(segment_X_test)
    print(f"Power {power}W - XGBoost MAE: {mean_absolute_error(segment_y_test, segment_y_pred)}")
    print(f"Power {power}W - XGBoost R2: {r2_score(segment_y_test, segment_y_pred)}")

# Visualization: Actual vs Predicted (for XGBoost; add RF if needed)
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual', color='black', linewidth=2)
plt.plot(y_pred_xgb, label='Predicted (XGBoost)', linestyle='--', color='green')
plt.title("Predicted vs Actual Min Temp (All Data)")
plt.xlabel("Sample Index")
plt.ylabel("Min Temp (K)")
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance for XGBoost
importances = best_xgb.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 5))
plt.bar(feature_names, importances)
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()

# Residual Plot for XGBoost
residuals = y_test - y_pred_xgb
plt.figure(figsize=(10, 5))
plt.scatter(range(len(residuals)), residuals, color='red', s=10)
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residuals of XGBoost Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.show()


# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load your cleaned dataset
df = pd.read_csv("master_cryocooler_data.csv")

# Feature Engineering: Add lagged features and transition detection
df['prev_min_temp'] = df['min_temp'].shift(1)
df['prev_power_input'] = df['power_input'].shift(1)
df['temp_change'] = df['min_temp'].diff().abs()
df['is_transition'] = (df['temp_change'] > 10).astype(int)
df = df.dropna()  # Drop rows with NaN from lagged features

# Convert 'time' to datetime and then to numeric (seconds since epoch)
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df['time_numeric'] = (df['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
df = df.dropna()  # Drop any rows where 'time' conversion failed

# Define features and target
X = df[['time_numeric', 'power_input', 'prev_power_input', 'chx_temp', 'outlet_temp', 'pr_regen', 'prev_min_temp', 'is_transition']]
y = df['min_temp']

# Split into training, validation, and testing sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 60% train, 20% val, 20% test

# Scale features for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Model 2: Optimized Random Forest with GridSearchCV and KFold
rf_param_grid = {'n_estimators': [150], 'max_depth': [6, 7]}
rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
rf_grid_search.fit(X_train, y_train)
best_rf = rf_grid_search.best_estimator_
y_pred_rf_train = best_rf.predict(X_train)
y_pred_rf_val = best_rf.predict(X_val)
y_pred_rf_test = best_rf.predict(X_test)

# Model 3: Optimized XGBoost with GridSearchCV and KFold
xgb_param_grid = {
    'n_estimators': [150],
    'max_depth': [4, 5],
    'learning_rate': [0.1],
    'subsample': [0.8, 1.0]
}
xgb_grid_search = GridSearchCV(xgb.XGBRegressor(random_state=42, objective='reg:squarederror'), xgb_param_grid, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
xgb_grid_search.fit(X_train, y_train)
best_xgb = xgb_grid_search.best_estimator_
y_pred_xgb_train = best_xgb.predict(X_train)
y_pred_xgb_val = best_xgb.predict(X_val)
y_pred_xgb_test = best_xgb.predict(X_test)

# Ensemble: Weighted Average Blending
# Optimize weights using validation set
from scipy.optimize import minimize

def objective(weights):
    w1, w2 = weights
    y_pred_val = w1 * y_pred_rf_val + w2 * y_pred_xgb_val
    return -r2_score(y_val, y_pred_val)  # Negative because minimize seeks minimum

initial_weights = [0.5, 0.5]
bounds = [(0, 1), (0, 1)]  # Weights between 0 and 1
constraints = {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1}  # Ensure weights sum to 1
result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
optimal_weights = result.x
print("Optimal weights (RF, XGBoost):", optimal_weights)

# Apply optimal weights to test set
y_pred_ensemble = optimal_weights[0] * y_pred_rf_test + optimal_weights[1] * y_pred_xgb_test

print("Best RF params:", rf_grid_search.best_params_)
print("Best XGBoost params:", xgb_grid_search.best_params_)
print("Linear MAE:", mean_absolute_error(y_test, y_pred_lr))
print("Linear R2:", r2_score(y_test, y_pred_lr))
print("RF MAE:", mean_absolute_error(y_test, y_pred_rf_test))
print("RF R2:", r2_score(y_test, y_pred_rf_test))
print("XGBoost MAE:", mean_absolute_error(y_test, y_pred_xgb_test))
print("XGBoost R2:", r2_score(y_test, y_pred_xgb_test))
print("Ensemble MAE:", mean_absolute_error(y_test, y_pred_ensemble))
print("Ensemble R2:", r2_score(y_test, y_pred_ensemble))

# Cross-validation with KFold for both RF and XGBoost
rf_cv_scores = cross_val_score(best_rf, X, y, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
xgb_cv_scores = cross_val_score(best_xgb, X, y, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
print("RF Cross-validated R2 (mean):", rf_cv_scores.mean())
print("RF Cross-validated R2 (std):", rf_cv_scores.std())
print("XGBoost Cross-validated R2 (mean):", xgb_cv_scores.mean())
print("XGBoost Cross-validated R2 (std):", xgb_cv_scores.std())

# Handle spikes by segmenting data (by power_input) for Ensemble
segments = df.groupby('power_input')
for power, segment_df in segments:
    segment_X = segment_df[['time_numeric', 'power_input', 'prev_power_input', 'chx_temp', 'outlet_temp', 'pr_regen', 'prev_min_temp', 'is_transition']]
    segment_y = segment_df['min_temp']
    segment_X_train, segment_X_test, segment_y_train, segment_y_test = train_test_split(segment_X, segment_y, test_size=0.2, random_state=42)
    segment_rf = RandomForestRegressor(**rf_grid_search.best_params_, random_state=42)
    segment_xgb = xgb.XGBRegressor(**xgb_grid_search.best_params_, random_state=42, objective='reg:squarederror')
    segment_rf.fit(segment_X_train, segment_y_train)
    segment_xgb.fit(segment_X_train, segment_y_train)
    segment_y_pred_rf = segment_rf.predict(segment_X_test)
    segment_y_pred_xgb = segment_xgb.predict(segment_X_test)
    segment_y_pred_ensemble = optimal_weights[0] * segment_y_pred_rf + optimal_weights[1] * segment_y_pred_xgb
    print(f"Power {power}W - Ensemble MAE: {mean_absolute_error(segment_y_test, segment_y_pred_ensemble)}")
    print(f"Power {power}W - Ensemble R2: {r2_score(segment_y_test, segment_y_pred_ensemble)}")

# Visualization: Actual vs Predicted (for Ensemble)
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual', color='black', linewidth=2)
plt.plot(y_pred_ensemble, label='Predicted (Ensemble)', linestyle='--', color='purple')
plt.title("Predicted vs Actual Min Temp (All Data)")
plt.xlabel("Sample Index")
plt.ylabel("Min Temp (K)")
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance for Ensemble (using XGBoost as proxy, since blending doesn't have direct importance)
importances = best_xgb.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 5))
plt.bar(feature_names, importances)
plt.title("Feature Importance (XGBoost as Proxy for Ensemble)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()

# Residual Plot for Ensemble
residuals = y_test - y_pred_ensemble
plt.figure(figsize=(10, 5))
plt.scatter(range(len(residuals)), residuals, color='red', s=10)
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residuals of Ensemble Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.show()


# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Load your cleaned dataset
df = pd.read_csv("master_cryocooler_data.csv")

# Feature Engineering: Add lagged features and transition detection
df['prev_min_temp'] = df['min_temp'].shift(1)
df['prev_power_input'] = df['power_input'].shift(1)
df['temp_change'] = df['min_temp'].diff().abs()
df['is_transition'] = (df['temp_change'] > 10).astype(int)
df = df.dropna()  # Drop rows with NaN from lagged features

# Convert 'time' to datetime and then to numeric (seconds since epoch)
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df['time_numeric'] = (df['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
df = df.dropna()  # Drop any rows where 'time' conversion failed

# Define features and target
X = df[['time_numeric', 'power_input', 'prev_power_input', 'chx_temp', 'outlet_temp', 'pr_regen', 'prev_min_temp', 'is_transition']]
y = df['min_temp']

# Split into training, validation, and testing sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 60% train, 20% val, 20% test

# Scale features for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Model 2: Optimized Random Forest with GridSearchCV and KFold
rf_param_grid = {'n_estimators': [150], 'max_depth': [6, 7]}
rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
rf_grid_search.fit(X_train, y_train)
best_rf = rf_grid_search.best_estimator_
y_pred_rf_test = best_rf.predict(X_test)

# Model 3: Optimized XGBoost with GridSearchCV and KFold
xgb_param_grid = {
    'n_estimators': [150],
    'max_depth': [5, 6],  # Increased from [4, 5] to improve performance
    'learning_rate': [0.1],
    'subsample': [0.8, 1.0]
}
xgb_grid_search = GridSearchCV(xgb.XGBRegressor(random_state=42, objective='reg:squarederror'), xgb_param_grid, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
xgb_grid_search.fit(X_train, y_train)
best_xgb = xgb_grid_search.best_estimator_
y_pred_xgb_test = best_xgb.predict(X_test)

# Model 4: Stacking Regressor
# Define base learners
base_learners = [
    ('rf', best_rf),
    ('xgb', best_xgb)
]

# Define meta-learner
meta_learner = LinearRegression()

# Build stacking model
stacking_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,  # cross-validation inside stacking
    passthrough=True  # pass original features along with base model predictions
)

# Fit stacking model
stacking_model.fit(X_train, y_train)

# Predictions
y_pred_stack_test = stacking_model.predict(X_test)

print("Best RF params:", rf_grid_search.best_params_)
print("Best XGBoost params:", xgb_grid_search.best_params_)
print("Linear MAE:", mean_absolute_error(y_test, y_pred_lr))
print("Linear R2:", r2_score(y_test, y_pred_lr))
print("RF MAE:", mean_absolute_error(y_test, y_pred_rf_test))
print("RF R2:", r2_score(y_test, y_pred_rf_test))
print("XGBoost MAE:", mean_absolute_error(y_test, y_pred_xgb_test))
print("XGBoost R2:", r2_score(y_test, y_pred_xgb_test))
print("Stacking MAE:", mean_absolute_error(y_test, y_pred_stack_test))
print("Stacking R2:", r2_score(y_test, y_pred_stack_test))

# Cross-validation with KFold for RF and XGBoost
rf_cv_scores = cross_val_score(best_rf, X, y, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
xgb_cv_scores = cross_val_score(best_xgb, X, y, cv=KFold(n_splits=15, shuffle=True, random_state=42), scoring='r2')
print("RF Cross-validated R2 (mean):", rf_cv_scores.mean())
print("RF Cross-validated R2 (std):", rf_cv_scores.std())
print("XGBoost Cross-validated R2 (mean):", xgb_cv_scores.mean())
print("XGBoost Cross-validated R2 (std):", xgb_cv_scores.std())

# Handle spikes by segmenting data (by power_input) for Stacking
segments = df.groupby('power_input')
for power, segment_df in segments:
    segment_X = segment_df[['time_numeric', 'power_input', 'prev_power_input', 'chx_temp', 'outlet_temp', 'pr_regen', 'prev_min_temp', 'is_transition']]
    segment_y = segment_df['min_temp']
    segment_X_train, segment_X_test, segment_y_train, segment_y_test = train_test_split(segment_X, segment_y, test_size=0.2, random_state=42)
    segment_stacking = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        passthrough=True
    )
    segment_stacking.fit(segment_X_train, segment_y_train)
    segment_y_pred_stack = segment_stacking.predict(segment_X_test)
    print(f"Power {power}W - Stacking MAE: {mean_absolute_error(segment_y_test, segment_y_pred_stack)}")
    print(f"Power {power}W - Stacking R2: {r2_score(segment_y_test, segment_y_pred_stack)}")

# Visualization: Actual vs Predicted (Stacking)
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual', color='black', linewidth=2)
plt.plot(y_pred_stack_test, label='Predicted (Stacking)', linestyle='--', color='blue')
plt.title("Predicted vs Actual Min Temp (Stacking)")
plt.xlabel("Sample Index")
plt.ylabel("Min Temp (K)")
plt.legend()
plt.grid(True)
plt.show()

# Residual Plot for Stacking
residuals = y_test - y_pred_stack_test
plt.figure(figsize=(10, 5))
plt.scatter(range(len(residuals)), residuals, color='green', s=10)
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residuals of Stacking Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.show()


# In[ ]:




