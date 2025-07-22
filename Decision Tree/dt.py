import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Load dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Fit the Decision Tree Regression model
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting a new result
predicted_salary = regressor.predict([[7.0]])
print("Predicted Salary for level 7.0:", predicted_salary)

# Visualizing the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
