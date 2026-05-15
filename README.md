# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries such as Pandas, Matplotlib, and KMeans from Scikit-learn.

2.Read the dataset "Mall_Customer.csv" and select the Annual Income and Spending Score columns for clustering.

3.Create the K-Means model with 5 clusters and train the model using the selected data.

4.Predict the cluster labels for all customer data points and find the cluster centers.

5.Plot the clusters and centroids using a scatter plot to visualize customer segmentation

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Mithun Kumar V
RegisterNumber:  212225040236
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customer.csv")

X = data.iloc[:, [3, 4]].values

kmeans = KMeans(n_clusters=5, random_state=0)

y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50)

plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200,
            marker='X')

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")

plt.show()  
```

## Output:
<img width="1064" height="641" alt="image" src="https://github.com/user-attachments/assets/259cb482-9431-488c-ba20-cdff6540e07b" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
