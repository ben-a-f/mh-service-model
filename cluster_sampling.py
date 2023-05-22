# This script implements a k-Means algorithm, and a function to generate new samples proportionally from each cluster.
# See cluster_algorithm_testing.py for tests of various approaches, and Cluster Model Inspection.ipynb for results.

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

# Import data, format and scale.
synthetic_patients = pd.read_csv("synthetic_patients.csv")
transform_cols = ["Age", "LoS", "DailyContacts"]
scaler = RobustScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(synthetic_patients[transform_cols]))
scaled_data.columns = transform_cols

# SET MANUAL PARAMETERS
# Set k, cluster, and extract labels and centroids.
k = 3
km = KMeans(n_clusters=k)
km.fit(scaled_data)

# Set ward parameters. UserWarning here is noted and accepted.
lower_bounds = np.array([18, 0, 0])
scaled_bounds = scaler.transform(lower_bounds.reshape(1, -1))
# END MANUAL PARAMETERS


# Generate truncated multivariate normal samples.
def truncated_mv_normal(mean, cov, size, lower_bounds, upper_bounds):
    samples = []
    while len(samples) < size:
        sample = np.random.multivariate_normal(mean=mean, cov=cov)
        if np.all(sample >= lower_bounds) and np.all(sample <= upper_bounds):
            samples.append(sample)
    return samples


# Generate total_n samples from the cluster model, using the model object kmeans.
def cluster_samples(kmeans, data, total_n, lower_bounds=np.array([-np.inf, -np.inf, -np.inf]),
                    upper_bounds=np.array([np.inf, np.inf, np.inf])):
    new_samples = pd.DataFrame(columns=[i for s in [[x for x in data.columns], ["Label"]] for i in s])
    # Get cluster proportions
    cluster_proportions = pd.Series(kmeans.labels_).value_counts() / np.sum(pd.Series(kmeans.labels_).value_counts())
    # For each cluster, define sample distribution and generate samples.
    for i in range(kmeans.n_clusters):
        n = round(total_n * cluster_proportions[i])
        centre = kmeans.cluster_centers_[i]
        points = data.loc[kmeans.labels_ == i]
        # Assign default covariance for single-sample clusters.
        if len(points) == 1:
            cov = np.eye(len(centre))
        else:
            cov = np.cov(points.T)
        # Sample new points and add to output dataframe.
        samples = pd.DataFrame(truncated_mv_normal(centre, cov, n, lower_bounds, upper_bounds), columns=data.columns)
        samples["Label"] = i
        new_samples = pd.concat([new_samples, samples], axis=0)
    return new_samples


# Generate samples.
scaled_samples = cluster_samples(km, scaled_data, 1000, scaled_bounds).reset_index()
synthetic_samples = pd.DataFrame(scaler.inverse_transform(scaled_samples[transform_cols]))
synthetic_samples.columns = transform_cols
synthetic_samples = pd.concat([synthetic_samples, scaled_samples["Label"]], axis=1)
