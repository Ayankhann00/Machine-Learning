import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values  # Position level column
y = dataset.iloc[:, -1].values    # Salary column
y = y.reshape(len(y), 1)          # Reshape for scaler

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fit SVR model
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y.ravel())  # y.ravel() to convert to 1D array for SVR

# Predicting a new result (e.g., position level 6.5)
predicted_salary = sc_y.inverse_transform(
    regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1)
)
print(f"Predicted salary for position level 6.5: {predicted_salary[0][0]:.2f}")

# Visualize the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red', label='Actual')
plt.plot(sc_X.inverse_transform(X), 
         sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), 
         color='blue', label='SVR Model')
plt.title('Truth or Bluff (SVR Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()
