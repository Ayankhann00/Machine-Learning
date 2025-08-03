import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Load dataset
dataset = pd.read_csv('Mall_Customers copy.csv')

# Select Annual Income and Spending Score
X = dataset.iloc[:, 3:].values

# Plot dendrogram
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# Apply Agglomerative Clustering
sc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
y_hc = sc.fit_predict(X)

# Visualize clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='magenta', label='Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='cyan', label='Cluster 5')
plt.title('Clusters of Customers Graph')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()