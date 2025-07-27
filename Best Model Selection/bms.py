# ML BEST MODEL
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Example: X = features, y = target
# For this example, create some random dummy data:
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
y = y.reshape(-1, 1)  # Reshape y for scaling

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features and target
Sc_X = StandardScaler()
Sc_y = StandardScaler()
X_train_scaled = Sc_X.fit_transform(X_train)
X_test_scaled = Sc_X.transform(X_test)
y_train_scaled = Sc_y.fit_transform(y_train).ravel()

# --- SVR Model ---
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)

# Predict and inverse transform
y_pred_svr_scaled = svr.predict(X_test_scaled).reshape(-1, 1)
y_pred_svr = Sc_y.inverse_transform(y_pred_svr_scaled)

# Calculate R-squared for SVR
r2_svr = r2_score(y_test, y_pred_svr)

# --- Decision Tree Model ---
dt = DecisionTreeRegressor(random_state=0)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
r2_dt = r2_score(y_test, y_pred_dt)

# --- Random Forest Model ---
rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(X_train, y_train.ravel())

y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)

# Print R-squared scores
print(f"R-squared (SVR): {r2_svr:.4f}")
print(f"R-squared (Decision Tree): {r2_dt:.4f}")
print(f"R-squared (Random Forest): {r2_rf:.4f}")

# Example prediction
new_data = [[10.0, 50.0, 1010.0, 60.0]]
scaled_input = Sc_X.transform(new_data)

svr_pred = Sc_y.inverse_transform(svr.predict(scaled_input).reshape(-1, 1))
dt_pred = dt.predict(new_data)
rf_pred = rf.predict(new_data)

print(f"SVR Prediction for new data: {svr_pred[0][0]:.2f}")
print(f"Decision Tree Prediction for new data: {dt_pred[0]:.2f}")
print(f"Random Forest Prediction for new data: {rf_pred[0]:.2f}")

import matplotlib.pyplot as plt

# 1. Scatter plot: Actual vs Predicted
plt.figure(figsize=(16, 5))

# SVR
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_svr, color='blue', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'SVR\nR² = {r2_svr:.2f}')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Decision Tree
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_dt, color='green', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Decision Tree\nR² = {r2_dt:.2f}')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Random Forest
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_rf, color='orange', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Random Forest\nR² = {r2_rf:.2f}')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()

# 2. Bar chart: R² comparison
models = ['SVR', 'Decision Tree', 'Random Forest']
scores = [r2_svr, r2_dt, r2_rf]

plt.figure(figsize=(6, 5))
plt.bar(models, scores, color=['blue', 'green', 'orange'])
plt.title('Model Comparison (R² Scores)')
plt.ylabel('R² Score')
plt.ylim(0, 1)
for i, v in enumerate(scores):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()
