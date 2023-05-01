# This script generates synthetic patient data.
# It is still a work in progress and the chosen parameters are largely arbitrary at present.

import numpy as np
import pandas as pd
from scipy.stats import beta, norm, lognorm


# Generate correlated random samples from a beta and two lognormal distributions.
# These are used to generate Age, Length of Stay (LoS) and Contact Frequency variables.
def correlated_samples(age_mean, age_var, los_mean, los_var, cf_mean, cf_var, corr_matx, min_age, max_age, n_samples):

    # Scale raw age to beta distribution parameters.
    beta_mean, beta_var = age_to_beta_samples(age_mean, age_var, min_age, max_age)

    # Calculate the beta distribution parameters alpha, beta.
    alpha_param = (((1 - beta_mean) / beta_var) - (1 / beta_mean)) * (beta_mean ** 2)
    beta_param = alpha_param * ((1 / beta_mean) - 1)

    # Calculate the los lognormal distribution parameters shape, scale.
    los_sd = np.sqrt(los_var)
    los_shape = np.sqrt(np.log(1 + (los_sd / los_mean) ** 2))
    los_mu = np.log(los_mean) - 0.5 * los_shape ** 2
    los_scale = np.exp(los_mu)

    # Calculate the contact frequency lognormal distribution parameters shape, scale.
    cf_sd = np.sqrt(cf_var)
    cf_shape = np.sqrt(np.log(1 + (cf_sd / cf_mean) ** 2))
    cf_mu = np.log(cf_mean) - 0.5 * cf_shape ** 2
    cf_scale = np.exp(cf_mu)

    # Set seed for reproducibility in testing.
    np.random.seed(42)

    # Generate multivariate normals with chosen correlation matrix.
    mv_normal = np.random.multivariate_normal(mean=[0, 0, 0], cov=corr_matx, size=n_samples)

    # Transform normals to uniform variables.
    mv_uniform = norm.cdf(mv_normal)

    # Transform uniforms to beta and lognormal.
    age_beta = beta.ppf(mv_uniform[:, 0], alpha_param, beta_param)
    los_lognorm = lognorm.ppf(mv_uniform[:, 1], s=los_shape, scale=los_scale).round()
    cf_lognorm = lognorm.ppf(mv_uniform[:, 2], s=cf_shape, scale=cf_scale)
    X = pd.DataFrame(np.column_stack((age_beta, los_lognorm, cf_lognorm)), columns=["Age", "LoS", "DailyContacts"])

    # Scale ages
    X["Age"] = beta_samples_to_age(min_age, max_age, X["Age"])

    return X


# Transform the specified age mean and variance from the interval [min_age, max_age] onto the interval [0,1].
def age_to_beta_samples(age_mean, age_var, min_age, max_age):
    beta_mean = (age_mean - min_age) / (max_age - min_age)
    beta_var = age_var / (max_age - min_age)
    return beta_mean, beta_var


# Transform random beta samples on [0,1] into samples on an interval [min_age, max_age].
def beta_samples_to_age(min_age, max_age, samples):
    age_samples = min_age + (max_age - min_age) * samples
    age_samples = age_samples.astype("int64")
    return age_samples


# Define several cohorts and generate.
# Low severity med stay, med severity short stay, emergency, low severity long stay, pfa entry
samples_n = [500, 200, 50, 100, 100]
age_means = [50, 60, 30, 40, 20]
age_vars = [3, 1.5, 5, 5, 1]
los_means = [42, 21, 7, 365, 84]
los_vars = [100, 28, 5, 1000, 100]
cf_means = [0.2, 0.5, 1.5, 0.035, 0.2]
cf_vars = [0.2, 0.2, 0.1, 0.1, 0.2]
corr_matxs = [[[1, 0.3, 0.3], [0.3, 1, -0.3], [0.3, -0.3, 1]],
              [[1, 0.5, 0.3], [0.5, 1, 0.0], [0.3, 0.0, 1]],
              [[1, 0.1, 0.4], [0.1, 1, 0.1], [0.4, 0.1, 1]],
              [[1, 0.1, 0.2], [0.1, 1, -0.3], [0.2, -0.3, 1]],
              [[1, 0.0, 0.0], [0.0, 1, -0.2], [0.0, -0.2, 1]]]

synthetic_patients = pd.DataFrame(columns=["Age", "LoS", "DailyContacts"])

for i in range(1):
    X = correlated_samples(age_means[i], age_vars[i], los_means[i], los_vars[i], cf_means[i], cf_vars[i], corr_matxs[i], 18, 100, samples_n[i])
    synthetic_patients = pd.concat([synthetic_patients, X])

synthetic_patients.reset_index(drop=True, inplace=True)

# Assign entry dates which are more heavily weighted towards winter months, depending on age and severity.
date_range = pd.date_range("2020-01-01", "2022-12-31")


def assign_entry_date(row, date_range):
    age_bracket = 0
    if 50 <= row["Age"] < 70:
        age_bracket = 1
    elif row["Age"] >= 70:
        age_bracket = 2

    severity_bracket = round(row["DailyContacts"]*10)

    month_weights = [10+age_bracket+severity_bracket,
                     10+age_bracket+severity_bracket,
                     10, 10, 10, 9, 9, 9, 10, 10, 10,
                     10+age_bracket+severity_bracket]
    date_probs = np.array([month_weights[date.month - 1] for date in date_range])
    date_probs = date_probs / sum(date_probs)

    entry_date = np.random.choice(date_range, p=date_probs)
    return entry_date


synthetic_patients["EntryDate"] = synthetic_patients.apply(assign_entry_date, axis=1, args=(date_range,))

synthetic_patients.to_csv("synthetic_patients.csv", index=False)
