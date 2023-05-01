import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

synthetic_patients = pd.read_csv("synthetic_patients.csv")
synthetic_patients["EntryDate"] = pd.to_datetime(synthetic_patients["EntryDate"], format="%Y-%m-%d")
synthetic_patients["EntryMonth"] = synthetic_patients["EntryDate"].dt.month

# TODO: Trial MinMaxScaler, StandardScaler, RobustScaler. Suspect high variance in LoS makes MinMax unsuitable.
transform_cols = ["Age", "LoS", "DailyContacts", "EntryMonth"]
scaler = RobustScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(synthetic_patients[transform_cols]))
scaled_data.columns = transform_cols

# TODO: Trial k-Means, Hierarchical and DBSCAN.
# First: DBSCAN
# Choose nearest n samples based on testing.
n = 2

# Create distance matrix.
dist = pd.DataFrame(squareform(pdist(scaled_data)),
                    columns=scaled_data.index,
                    index=scaled_data.index)

# Get nth nearest neighbour for all points, sort ascending, and find 'elbow' by eye.
nth_dist = dist.apply(lambda row: row.nsmallest(n).values[-1], axis=1).sort_values().reset_index(drop=True)
fig, ax = plt.subplots()
ax.plot(nth_dist)

# Choose epsilon which defines the neighbourhood size.
epsilon = 1.5

# Apply DBSCAN and check results.
db = DBSCAN(eps=epsilon, min_samples=n)
db.fit(scaled_data)
labels = pd.Series(db.labels_)

# Silhouette score [-1,1] is 'goodness-of-fit' for clustering:
    # High score: Clusters are far apart and clearly distinct.
    # Zero score: Distance between clusters is not significant.
    # Negative score: Clusters are assigned wrong.
print("Labels: ", labels.unique())
print("Silhouette Score: ", silhouette_score(scaled_data, labels))

