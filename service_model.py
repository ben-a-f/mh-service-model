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
ward1 = simpy.Resource(env, capacity=2)
ward2 = simpy.Resource(env, capacity=3)
ward1.name = "ward1"
ward2.name = "ward2"
wards = [ward1, ward2]


# Define a patient.
class Patient(object):
    # Instantiate a patient and generate characteristics.
    def __init__(self, env, id_num, resource_req, req_index):
        self.env = env
        self.id = id_num
        self.cluster = np.random.choice(range(len(global_props)), 1, p=global_props)[0]
        self.patient_characteristics = truncated_mv_normal(global_centres[self.cluster],
                                                           global_covs[self.cluster],
                                                           global_scaled_bounds)
        self.action = env.process(self.enter_service(env, resource_req, req_index))

    def enter_service(self, env, resource_req, req_index):
        # TODO: Automate reading LoS from characteristics.
        los = 5
        patient_name = "P" + str(self.id)
        print(f"{patient_name} occupied {wards[req_index].name} at time {env.now}")
        yield env.timeout(los)
        wards[req_index].release(resource_req)
        print(f"{patient_name} released {wards[req_index].name} at time {env.now}")

        # TODO: Allow for sequence of resources (i.e. move to next ward).



# Referral process creates new patients at each time step and assigns them to an available ward.
def daily_referrals(env, resources):
    id_start = 0
    while True:
        # TODO: Instantiate a random number patients using global entry rate.
        #       Execute the ward finding process below for each.
        # num_instances = np.random.randint(1, 3)

        # TODO: Instantiate patients first so the possible wards can be subset based on their attributes.
        #       Then trigger the enter_service(), rather than triggering on instantiation.
        # Find a free resource and send the patient there.
        reqs = [ward.request() for ward in wards]
        req_index = 0
        # returns dictionary of triggered events.
        result = yield simpy.AnyOf(env, reqs)
        # Find the successful request
        resourceList = list(result.keys())
        resource = resourceList[0]

        # Case 1: Multiple requests got filled, need to release all others
        if len(resourceList) > 1:
            for i in range(len(reqs)):
                if reqs[i] != resource:
                    wards[i].release(reqs[i])
                else:
                    req_index = i
        # Case 2: Only one request was filled, need to cancel the others
        else:
            for i in range(len(reqs)):
                if reqs[i] not in resourceList:
                    reqs[i].cancel()
                else:
                    req_index = i

        Patient(env, id_start, reqs[req_index], req_index)
        id_start += 1

        # Wait for the next time step
        yield env.timeout(1)


# Create an instance of the event process and start it
env.process(daily_referrals(env, wards))

# Run the simulation
env.run(until=10)  # Change the simulation duration as needed

# TODO: Extract occupancies over full duration.