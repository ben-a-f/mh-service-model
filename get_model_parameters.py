# This script contains the functions needed to retrieve statistics from a dataset for the service model.

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# TODO: Add functionality to give additional weight to most recent n months by oversampling.

# Cluster a patient dataset and return the cluster ratios, means and covariances to be passed to the model.
def get_cluster_parameters(scaled_data, k):
    # Fit kMeans.
    km = KMeans(n_clusters=k)
    km.fit(scaled_data)

    # Retrieve cluster details.
    cluster_proportions = pd.Series(km.labels_).value_counts() / np.sum(pd.Series(km.labels_).value_counts())
    cluster_centres = km.cluster_centers_
    cluster_covariances = []
    for i in range(km.n_clusters):
        points = scaled_data.loc[km.labels_ == i]
        if len(points) == 1:
            cov = np.eye(len(cluster_centres[i]))
        else:
            cov = np.cov(points.T)
        cluster_covariances.append(cov)
    cluster_covariances = np.array(cluster_covariances)
    return cluster_proportions, cluster_centres, cluster_covariances

# Get the average daily entry rate.
def get_referral_rate(entry_dates):
    entry_dates = pd.to_datetime(entry_dates).value_counts()
    all_dates = pd.DataFrame(index=pd.date_range(entry_dates.index.min(), entry_dates.index.max()))
    referral_rate = all_dates.merge(entry_dates, how="left", left_index=True, right_index=True).fillna(0).mean()
    return referral_rate[0]

# Generate synthetic patient characteristics from a given mean and covariance, within specified bounds.
def truncated_mv_normal(scaler, mean, cov, lower_bounds=None):
    # Assuming all parameters should be non-negative.
    if lower_bounds is None:
        lower_bounds = np.repeat(0, len(mean))

    p = np.random.multivariate_normal(mean=mean, cov=cov)
    if np.any(np.less(p, lower_bounds)):
        p = truncated_mv_normal(mean, cov, lower_bounds)
    p = scaler.inverse_transform(p.reshape(1, -1))
    p[0, 0:2] = p[0, 0:2].round()
    return p

# Count historical occupancy from referral dates, discharge dates, and ward names.
def get_historical_occupancy(data):
    occupancy = pd.DataFrame(index=pd.data_range(data["ReferralDate"].min(), data["DischargeDate"].max()))
    # TODO: Count occupancy for each ward.
    return occupancy
