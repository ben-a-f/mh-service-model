# This script contains the early WIP of the SimPy model that describes an inpatient mental health service.

import simpy
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

np.random.seed(42)

# Import data, format and scale.
synthetic_patients = pd.read_csv("synthetic_patients.csv")
transform_cols = ["Age", "LoS", "DailyContacts"]
scaler = RobustScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(synthetic_patients[transform_cols]))
scaled_data.columns = transform_cols

# TODO: Replace with read-in parameters.
# Set some initial parameters as global variables by hand for testing.
# Cluster probabilities
global_props = [0.874, 0.114, 0.012]
# Age, LoS and DailyContacts means & cov matx for each cluster.
global_centres = [[-0.02348025,  0.03872558,  0.09976081],
                  [ 0.54347826, -0.08974359, 15.49924954],
                  [ 0.2532418, -0.34412955,  4.16255464]]
global_covs = [[[ 4.59758542e-01,  1.33754795e-01,  1.11567438e-01],
                [ 1.33754795e-01,  4.87558287e-01, -4.43869321e-02],
                [ 1.11567438e-01, -4.43869321e-02,  3.09340565e-01]],
               [[ 1.82041588e-01,  5.18394649e-02,  3.16903395e-01],
                [ 5.18394649e-02,  2.44773176e-01, -1.26607217e+00],
                [ 3.16903395e-01, -1.26607217e+00,  4.85179220e+01]],
               [[ 4.22166001e-01,  2.24155707e-01,  4.80202957e-02],
                [ 2.24155707e-01,  5.04248788e-01, -9.56266408e-02],
                [ 4.80202957e-02, -9.56266408e-02,  3.10893071e+00]]]
# Patient referral rate
global_entry_rate = 0.5
# Min age to be permitted is 18.
global_bounds = np.array([18, 0, 0])
global_scaled_bounds = scaler.transform(global_bounds.reshape(1, -1))
# Ward requirements: None, Age > 65+, LoS <7
global_ward_reqs_low = [[0, 0, 0], # Ward 1
                        [65, 0, 0], # Ward 2
                        [0, 0, 0]] # Ward 3
global_ward_reqs_high = [[np.inf, np.inf, np.inf],
                         [np.inf, np.inf, np.inf],
                         [np.inf, 14, np.inf]]


# Generate synthetic patient characteristics from a given mean and cov (which are based on a specific cluster).
# Age, LoS, DailyContacts
def truncated_mv_normal(mean, cov, lower_bounds=None):
    if lower_bounds is None:
        lower_bounds = np.repeat(-np.inf, len(mean))

    p = np.random.multivariate_normal(mean=mean, cov=cov)
    if np.any(np.less(p, lower_bounds)):
        p = truncated_mv_normal(mean, cov, lower_bounds)
    p = scaler.inverse_transform(p.reshape(1, -1))
    p[0, 0:2] = p[0, 0:2].round()
    return p


# Create the simulation environment
env = simpy.Environment()
ward1 = simpy.Resource(env, capacity=10)
ward2 = simpy.Resource(env, capacity=10)
ward3 = simpy.Resource(env, capacity=5)
ward1.name = "General Ward"
ward2.name = "Senior Ward"
ward3.name = "Short Stay"
wards = [ward1, ward2, ward3]
occupancy = {ward1.name: [],
             ward2.name: [],
             ward3.name: []}


# Define a patient.
class Patient(object):
    # Instantiate a patient and generate characteristics if not supplied.
    def __init__(self, env, id_num, cluster=None, characteristics=None):
        self.env = env
        self.id = id_num
        if cluster is None:
            self.cluster = np.random.choice(range(len(global_props)), 1, p=global_props)[0]
        else:
            self.cluster = cluster
        if characteristics is None:
            self.patient_characteristics = truncated_mv_normal(global_centres[self.cluster],
                                                               global_covs[self.cluster],
                                                               global_scaled_bounds)
        else:
            self.patient_characteristics = characteristics

    # TODO: Remove req_index and print statements once testing is done.
    def enter_service(self, env, resource_req, req_index, suitable_wards):
        los = self.patient_characteristics[0, 1]
        patient_name = "P" + str(self.id)
        # print(f"{patient_name} occupied {suitable_wards[req_index].name} at time {env.now}")
        yield env.timeout(los)
        wards[req_index].release(resource_req)
        # print(f"{patient_name} released {suitable_wards[req_index].name} at time {env.now}")

        # TODO: Allow for sequence of resources (i.e. move to next ward).
        #       Update: Have added option to instantiate with known characteristics.
        #       Follow-up stays can be instantiated as new patient with same ID.



# Referral process creates new patients at each time step and assigns them to an available ward.
def daily_referrals(env):
    id_start = 0
    while True:
        # Generate number of new referrals for this time step (time step: 1 day)
        num_referrals = np.random.poisson(global_entry_rate)

        for j in range(num_referrals):
            # Instantiate patient and check for suitable wards based on their characteristics.
            p = Patient(env, id_start)
            suitable_wards = []
            for i in range(len(global_ward_reqs_low)):
                if np.any(np.less(p.patient_characteristics, global_ward_reqs_low[i])) or np.any(np.greater(p.patient_characteristics, global_ward_reqs_high[i])):
                    continue
                else:
                    suitable_wards.append(wards[i])

            # Find a free resource and send the patient there.
            reqs = [ward.request() for ward in suitable_wards]
            req_index = 0
            # returns dictionary of triggered events.
            # TODO: I think this line is causing the process to wait until one is fulfilled, instead of continuing on.
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

            # Send patient to ward.
            env.process(p.enter_service(env, reqs[req_index], req_index, suitable_wards))
            id_start += 1

        # Record occupancies.
        print(env.now)
        if env.now >= 20:
            print("Stop!")
        for ward in wards:
            occupancy[ward.name].append((env.now, ward.count))
        # Wait for the next time step
        yield env.timeout(1)


# Create an instance of the event process and start it
env.process(daily_referrals(env))

# Run the simulation
env.run(until=100)

# TODO: Extract occupancies over full duration.
#       Export data.
#       Export visualisations.