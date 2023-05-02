import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import numpy as np

synthetic_patients = pd.read_csv("synthetic_patients.csv")
synthetic_patients["EntryDate"] = pd.to_datetime(synthetic_patients["EntryDate"], format="%Y-%m-%d")
synthetic_patients["EntryMonth"] = synthetic_patients["EntryDate"].dt.month

# TODO: Trial MinMaxScaler, StandardScaler, RobustScaler. Suspect high variance in LoS makes MinMax unsuitable.
transform_cols = ["Age", "LoS", "DailyContacts", "EntryMonth"]
scaler = RobustScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(synthetic_patients[transform_cols]))
scaled_data.columns = transform_cols

# DBSCAN
# Identify values of epsilon (neighbourhood size) for various core point definitions.
n = 6

# Create distance matrix.
dist = pd.DataFrame(squareform(pdist(scaled_data)),
                    columns=scaled_data.index,
                    index=scaled_data.index)

# Get nth nearest neighbour for all points, sort ascending, and find 'elbow' by eye.
nth_dist = dist.apply(lambda row: row.nsmallest(n).values[-1], axis=1).sort_values().reset_index(drop=True)
fig, ax1 = plt.subplots()
ax1.plot(nth_dist)


# data = scaled_data
# eps = 1.5
# min_samples = 3
def apply_DBSCAN(data, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(data)
    labels = pd.Series(db.labels_)

    n_clusters = len(labels.unique())
    silhouette = silhouette_score(data, labels)

    centroids = []
    for label in labels.unique():
        if label != -1:
            centroid = np.array(np.mean(data[labels == label], axis=0))
            centroids.append(centroid)
    centroids = np.array(centroids)
    sse = np.sum(np.array((data - centroids[labels])) ** 2)

    return n_clusters, silhouette, sse


min_samples = [2, 3, 4, 5, 6]
eps = [1.05, 1.5, 1.6, 1.7, 1.7]
num_clusters = []
silhouette_scores = []
sse_scores = []

for i in range(len(min_samples)):
    n, sl, sse = apply_DBSCAN(scaled_data, eps[i], min_samples[i])
    num_clusters.append(n)
    silhouette_scores.append(sl)
    sse_scores.append(sse)

results_DBSCAN = pd.DataFrame(list(zip(min_samples, eps, num_clusters, silhouette_scores, sse_scores)),
                              columns=["min_samples", "epsilon", "num_clusters", "silhouette_scores", "sse_scores"])

# So let's use n_samples = 3 and eps = 1.6.

# K-Means
silhouette_scores = []
sse_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k)
    km.fit(scaled_data)
    labels = pd.Series(km.labels_)
    silhouette_scores.append(silhouette_score(scaled_data, labels))
    sse_scores.append(km.inertia_)

results_kMeans = pd.DataFrame(list(zip(range(2, 11), silhouette_scores, sse_scores)),
                              columns=["num_clusters", "silhouette_scores", "sse_scores"])

fig, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(results_kMeans["num_clusters"], results_kMeans["sse_scores"])
ax2.set_xlabel('k-value')
ax2.set_ylabel('error')

# k-Means gives much lower SSE with comparable silhouette scores than DBSCAN.

# Agglomerative
# Calculate linkage matrix
Z = linkage(scaled_data, method='ward')
# Generate dendrogram
fig, ax3 = plt.subplots(figsize=(10, 5))
dendrogram(Z, ax=ax3)
plt.show()
# Looks like 3 may be a good number, but let's add in some extras to be thorough.
max_clusters = [2, 3, 4, 5, 6]
silhouette_scores = []
sse_scores = []
for t in max_clusters:
    # We use -1 because fcluster labels from 1 and this makes indexing more intuitive.
    labels = pd.Series(fcluster(Z, t=t, criterion='maxclust')) - 1
    silhouette_scores.append(silhouette_score(scaled_data, labels))

    centroids = []
    for label in labels.unique():
        centroid = np.array(np.mean(scaled_data[labels == label], axis=0))
        centroids.append(centroid)
    centroids = np.array(centroids)
    sse = np.sum(np.array((scaled_data - centroids[labels])) ** 2)
    sse_scores.append(sse)

results_agg = pd.DataFrame(list(zip(max_clusters, silhouette_scores, sse_scores)),
                           columns=["num_clusters", "silhouette_scores", "sse_scores"])

# TODO: Standardise the coding for the three clustering alogrithms.
#  Make a function for calculating sse.
#  Remove apply_DBSCAN() and just use a loop.

# TODO: Compare the approaches.