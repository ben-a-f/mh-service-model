# This script implements a k-Means algorithm, and a function to generate new samples proportionally from each cluster.
# See patient_clusters.py for clustering algorithm testing, and Cluster Model Inspection.ipynb for results.

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

# Import data, format and scale.
synthetic_patients = pd.read_csv("synthetic_patients.csv")
synthetic_patients["EntryDate"] = pd.to_datetime(synthetic_patients["EntryDate"], format="%Y-%m-%d")
synthetic_patients["EntryMonth"] = synthetic_patients["EntryDate"].dt.month
transform_cols = ["Age", "LoS", "DailyContacts", "EntryMonth"]
scaler = RobustScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(synthetic_patients[transform_cols]))
scaled_data.columns = transform_cols

# Set k, cluster, and extract labels and centroids.
k = 3
km = KMeans(n_clusters=k)
km.fit(scaled_data)


# Generate total_n samples from the cluster model, using the model object kmeans.
def cluster_samples(kmeans, data, total_n):
    new_samples = pd.DataFrame(columns=[i for s in [[x for x in data.columns], ["Label"]] for i in s])
    # Get cluster proportions
    cluster_proportions = pd.Series(kmeans.labels_).value_counts() / np.sum(pd.Series(kmeans.labels_).value_counts())
    # Define (approximate) sample distribution and draw from it for each cluster.
    for i in range(kmeans.n_clusters):
        # Calculate centre and covariance matrix.
        centre = kmeans.cluster_centers_[i]
        points = data.loc[kmeans.labels_ == i]
        cov = np.cov(points.T)
        # Sample new points and add to output dataframe.
        n = round(total_n * cluster_proportions[i])
        samples = pd.DataFrame(np.random.multivariate_normal(mean=centre, cov=cov, size=n), columns=data.columns)
        samples["Label"] = i
        new_samples = pd.concat([new_samples, samples], axis=0)
    return new_samples


synthetic_samples = cluster_samples(km, scaled_data, 10000)








