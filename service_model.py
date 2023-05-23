# This script contains the early WIP of the SimPy model that describes an inpatient mental health service.

import simpy
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Import data, format and scale.
synthetic_patients = pd.read_csv("synthetic_patients.csv")
transform_cols = ["Age", "LoS", "DailyContacts"]
scaler = RobustScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(synthetic_patients[transform_cols]))
scaled_data.columns = transform_cols

# Set some initial parameters as global variables by hand for testing.
global_props = [0.874, 0.114, 0.012]
global_centres = [[-0.02348025,  0.03872558,  0.09976081],
                  [ 0.54347826, -0.08974359, 15.49924954],
                  [ 0.2532418 , -0.34412955,  4.16255464]]
global_covs = [[[ 4.59758542e-01,  1.33754795e-01,  1.11567438e-01],
                [ 1.33754795e-01,  4.87558287e-01, -4.43869321e-02],
                [ 1.11567438e-01, -4.43869321e-02,  3.09340565e-01]],
               [[ 1.82041588e-01,  5.18394649e-02,  3.16903395e-01],
                [ 5.18394649e-02,  2.44773176e-01, -1.26607217e+00],
                [ 3.16903395e-01, -1.26607217e+00,  4.85179220e+01]],
               [[ 4.22166001e-01,  2.24155707e-01,  4.80202957e-02],
                [ 2.24155707e-01,  5.04248788e-01, -9.56266408e-02],
                [ 4.80202957e-02, -9.56266408e-02,  3.10893071e+00]]]
global_entry_rate = 0.5
global_bounds = np.array([18, 0, 0])
global_scaled_bounds = scaler.transform(global_bounds.reshape(1, -1))


# Generate synthetic patient characteristics from a given mean and cov (which are based on a specific cluster).
def truncated_mv_normal(mean, cov, lower_bounds=None):
    if lower_bounds is None:
        lower_bounds = np.repeat(-np.inf, len(mean))

    p = np.random.multivariate_normal(mean=mean, cov=cov)
    if np.any(p < lower_bounds):
        p = truncated_mv_normal(mean, cov, lower_bounds)
    scaler.inverse_transform(p.reshape(1, -1))
    return p


# Create the simulation environment
env = simpy.Environment()
# ward = simpy.Resource(env, capacity=20)


# Define a patient.
class Patient(object):
    # Start the instantiate process every time an instance is created.
    def __init__(self, env):
        self.env = env
        self.action = env.process(self.instantiate())

    # Instantiate a patient with synthetic data.
    def instantiate(self):
        while True:
            cluster = np.random.choice(range(len(global_props)), 1, p=global_props)[0]
            patient_characteristics = truncated_mv_normal(global_centres[cluster],
                                                          global_covs[cluster],
                                                          global_scaled_bounds)
            # TODO: Identify appropriate resources.
            # with ward.request() as req
            #   yield req
            #   yield env.timeout(los)

            # TODO: Allow for sequence of resources (i.e. move to next ward).



# Create the event process
def daily_referrals(env):
    while True:
        # TODO: Instantiate a random number patients using global entry rate.
        num_instances = np.random.randint(1, 3)
        for _ in range(num_instances):
            Patient(env)

        # Wait for the next time step
        yield env.timeout(1)  # Change the time step as needed


# Create an instance of the event process and start it
env.process(daily_referrals(env))

# Run the simulation
env.run(until=10)  # Change the simulation duration as needed

# TODO: Extract occupancies over full duration.