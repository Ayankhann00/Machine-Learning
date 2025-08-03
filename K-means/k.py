import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, 3:].values


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Graph')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
ykmeans = kmeans.fit_predict(X)


plt.scatter(X[ykmeans == 0, 0], X[ykmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[ykmeans == 1, 0], X[ykmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[ykmeans == 2, 0], X[ykmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[ykmeans == 3, 0], X[ykmeans == 3, 1], s=100, c='magenta', label='Cluster 4')
plt.scatter(X[ykmeans == 4, 0], X[ykmeans == 4, 1], s=100, c='cyan', label='Cluster 5')
plt.title('Clusters of Customers Graph')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()