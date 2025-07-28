# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn tools for preprocessing, model, and evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Display first few rows
print("Dataset preview:")
print(dataset.head())

# Split features and target
X = dataset.iloc[:, :-1].values  # Assuming last column is target
y = dataset.iloc[:, -1].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict a sample input [Age=30, EstimatedSalary=41000]
sample_prediction = classifier.predict(sc.transform([[30, 41000]]))
print(f"\nPrediction for [30, 41000]: {sample_prediction[0]}")

# Predict on test set
y_pred = classifier.predict(X_test)

# Combine predicted vs actual results
results = np.concatenate(
    (y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1)
print("\nPredicted vs Actual (y_pred | y_test):")
print(results)

# Confusion matrix and accuracy
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy:.2f}")

# Visualize confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
