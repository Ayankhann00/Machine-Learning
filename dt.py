import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
print(dataset.head())

# Separate features and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the Decision Tree model
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

# Predict a single new result
new_prediction = classifier.predict(sc.transform([[30, 41000]]))
print(f"Prediction for age 30 and salary 41000: {new_prediction[0]}")

# Predict the test set results
y_pred = classifier.predict(X_test)

# Compare predictions with actual values
comparison = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), axis=1)
print("Predictions vs Actual values:\n", comparison)

# Confusion matrix and accuracy
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print(f"Accuracy: {ac:.2f}")
