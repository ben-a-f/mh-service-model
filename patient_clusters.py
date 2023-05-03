# This file trials two scaling methods and four clustering algorithms on the synthetic patient data.
# Scaling: RobustScaler(), StandardScaler()
# Clustering: DBSCAN, k-Means, Agglomerative (Hierarchical), Gaussian Mixture Model
# The results are outputted in .csv format and inspected in a separate script, cluster_model_inspection.py

import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import numpy as np

synthetic_patients = pd.read_csv("synthetic_patients.csv")
synthetic_patients["EntryDate"] = pd.to_datetime(synthetic_patients["EntryDate"], format="%Y-%m-%d")
synthetic_patients["EntryMonth"] = synthetic_patients["EntryDate"].dt.month

# Note: After some testing, the StandardScaler() is less effective than the RobustScaler().
# The sse scores are comparable but the silhouette scores are worse, suggesting that the optimal number of clusters
# with either method is similar, but that the clusters defined with StandardScaler() data are less distinct.
transform_cols = ["Age", "LoS", "DailyContacts", "EntryMonth"]
scaler = RobustScaler()
# scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(synthetic_patients[transform_cols]))
scaled_data.columns = transform_cols


# Function to calculate the Sum Squared Error (sse) of a given clustering.
def calculate_sse(labels, data):
    centroids = []
    sse = 0
    for i in labels.unique():
        cluster_points = np.array(data[labels == i])
        center = np.array(np.mean(cluster_points, axis=0))
        centroids.append(center)
        sse += np.sum((cluster_points - center) ** 2)
    return sse

# DBSCAN
# Visually inspect various choices of min_samples, n.
# Defines the min number of samples in a neighbourhood to define a core point.
n = 8
# Create distance matrix.
dist = pd.DataFrame(squareform(pdist(scaled_data)),
                    columns=scaled_data.index,
                    index=scaled_data.index)
# Get distance to nth neighbour (sorted ascending) and find 'elbow' by eye to identify suitable epsilon values.
nth_dist = dist.apply(lambda row: row.nsmallest(n).values[-1], axis=1).sort_values().reset_index(drop=True)
fig, ax1 = plt.subplots()
ax1.plot(nth_dist)

# Apply algorithm for a range of parameters.
min_samples = [2, 3, 4, 5, 6, 7, 8]
eps = [0.95, 1.1, 1.3, 1.5, 1.6, 1.7, 1.7]
num_clusters = []
silhouette_scores = []
sse_scores = []

for i in range(len(min_samples)):
    # Apply algorithm and extract cluster labels.
    db = DBSCAN(eps=eps[i], min_samples=min_samples[i])
    db.fit(scaled_data)
    labels = pd.Series(db.labels_)
    # Extract number of clusters.
    num_clusters.append(len(labels.unique()))
    # Get silhouette and sse scores.
    silhouette_scores.append(silhouette_score(scaled_data, labels))
    sse_scores.append(calculate_sse(labels, scaled_data))
# Record parameters and results in dataframe.
results_DBSCAN = pd.DataFrame(list(zip(min_samples, eps, num_clusters, silhouette_scores, sse_scores)),
                              columns=["min_samples", "epsilon", "num_clusters", "silhouette_scores", "sse_scores"])
results_DBSCAN["algorithm"] = "DBSCAN"


# K-Means
# Apply algorithm for a range of parameters.
k_range = [2, 3, 4, 5, 6, 7, 8]
silhouette_scores = []
sse_scores = []
for k in k_range:
    # Apply algorithm and extract cluster labels.
    km = KMeans(n_clusters=k)
    km.fit(scaled_data)
    labels = pd.Series(km.labels_)
    # Get silhouette and sse scores. k-means has built in sse calculation through .inertia_
    silhouette_scores.append(silhouette_score(scaled_data, labels))
    sse_scores.append(km.inertia_)
# Record parameters and results in dataframe.
results_kMeans = pd.DataFrame(list(zip(k_range, silhouette_scores, sse_scores)),
                              columns=["num_clusters", "silhouette_scores", "sse_scores"])
results_kMeans["algorithm"] = "kmeans"


# Agglomerative
# Calculate linkage matrix
Z = linkage(scaled_data.to_numpy(), method='ward')
# Generate dendrogram for inspection
fig, ax3 = plt.subplots(figsize=(10, 5))
dendrogram(Z, ax=ax3)
plt.show()

# Apply algorithm for a range of parameters.
max_clusters = [2, 3, 4, 5, 6, 7, 8]
silhouette_scores = []
sse_scores = []
for t in max_clusters:
    # We use -1 because fcluster labels from 1 by default and this makes indexing more intuitive.
    labels = pd.Series(fcluster(Z, t=t, criterion='maxclust')) - 1
    silhouette_scores.append(silhouette_score(scaled_data, labels))
    sse_scores.append(calculate_sse(labels, scaled_data))
# Record parameters and results in dataframe.
results_agg = pd.DataFrame(list(zip(max_clusters, silhouette_scores, sse_scores)),
                            columns=["num_clusters", "silhouette_scores", "sse_scores"])
results_agg["algorithm"] = "agglomerative"


# Gaussian Mixture Models
# Apply algorithm for a range of parameters.
n_range = [2, 3, 4, 5, 6, 7, 8]
silhouette_scores = []
sse_scores = []
for n in n_range:
    # Apply algorithm and extract cluster labels.
    gmm = GaussianMixture(n_components=n)
    gmm.fit(scaled_data)
    labels = pd.Series(gmm.predict(scaled_data))
    # Get silhouette and sse scores. k-means has built in sse calculation through .inertia_
    silhouette_scores.append(silhouette_score(scaled_data, labels))
    sse_scores.append(calculate_sse(labels, scaled_data))
# Record parameters and results in dataframe.
results_gmm = pd.DataFrame(list(zip(n_range, silhouette_scores, sse_scores)),
                           columns=["num_clusters", "silhouette_scores", "sse_scores"])
results_gmm["algorithm"] = "gmm"

results = pd.concat([results_gmm, results_kMeans, results_agg, results_DBSCAN]).reset_index(drop=True)
results["scaler"] = "robust"
results.to_csv("robust_scaler_clustering.csv", index=False)
# results["scaler"] = "standard"
# results.to_csv("standard_scaler_clustering.csv", index=False)