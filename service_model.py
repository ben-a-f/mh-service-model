# This WIP script contains the SimPy model that describes an inpatient mental health service.
# There is a "Set manual parameters." section that needs to be completed.

import simpy
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from get_model_parameters import get_cluster_parameters
from get_model_parameters import get_referral_rate
from get_model_parameters import truncated_mv_normal

# TODO: Remove once testing is over.
np.random.seed(42)

# TODO: Check if we need to remove NaNs from LoS column before/after scaling
#       (but keep them in the raw data for occupancy func).
# Import data, format and scale.
synthetic_patients = pd.read_csv("synthetic_patients.csv")
transform_cols = ["Age", "LoS", "DailyContacts"]
scaler = RobustScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(synthetic_patients[transform_cols]))
scaled_data.columns = transform_cols

# Get model parameters
global_props, global_centres, global_covs = get_cluster_parameters(scaled_data, 3)
global_entry_rate = get_referral_rate(synthetic_patients["EntryDate"])

# Create the simulation environment
env = simpy.Environment()
ward1 = simpy.Resource(env, capacity=10)
ward2 = simpy.Resource(env, capacity=10)
ward3 = simpy.Resource(env, capacity=5)
ward1.name = "General Ward"
ward2.name = "Senior Ward"
ward3.name = "Long Stay"
# This vector indirectly acts as a priority.
wards = [ward3, ward2, ward1]

# TODO: Instantiate occupancy with existing patients.
# Dicts to track occupancy and queues.
occupancy = {ward1.name: [],
             ward2.name: [],
             ward3.name: []}
queue = {ward1.name: [],
         ward2.name: [],
         ward3.name: []}
queue_count = {ward1.name: 0,
               ward2.name: 0,
               ward3.name: 0}

## Set manual parameters.
# Minimum patient Age is set to be 18. LoS and DailyContacts are >=0.
global_bounds = np.array([18, 0, 0])
global_scaled_bounds = scaler.transform(global_bounds.reshape(1, -1))
# Set ward specialisations manually: General Purpose, Age >65, and LoS >56.
global_ward_reqs_low = {ward1.name: [0, 0, 0],
                        ward2.name: [65, 0, 0],
                        ward3.name: [0, 56, 0]}
global_ward_reqs_high = {ward1.name: [np.inf, np.inf, np.inf],
                         ward2.name: [np.inf, np.inf, np.inf],
                         ward3.name: [np.inf, np.inf, np.inf]}
##

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
            self.patient_characteristics = truncated_mv_normal(scaler,
                                                               global_centres[self.cluster],
                                                               global_covs[self.cluster],
                                                               global_scaled_bounds)
        else:
            self.patient_characteristics = characteristics
        self.action = env.process(self.choose_service(env))

    # Find possible services for this patient and accept first available space.
    def choose_service(self, env):
        suitable_wards = []
        for ward in wards:
            if np.all(np.greater_equal(self.patient_characteristics, global_ward_reqs_low[ward.name])) and np.all(
                    np.less_equal(self.patient_characteristics, global_ward_reqs_high[ward.name])):
                suitable_wards.append(ward)

        # Find a free resource and send the patient there.
        reqs = [ward.request() for ward in suitable_wards]
        # Add each request to the queue count (note: this double counts for multiple possible wards).
        for ward in suitable_wards:
            queue_count[ward.name] += 1
        req_index = 0
        # Returns dictionary of triggered events.
        result = yield simpy.AnyOf(env, reqs)
        # Find the successful request.
        resourceList = list(result.keys())
        resource = resourceList[0]

        # Case 1: Multiple requests got filled, need to release all others
        if len(resourceList) > 1:
            for i in range(len(reqs)):
                if reqs[i] != resource:
                    suitable_wards[i].release(reqs[i])
                    queue_count[suitable_wards[i].name] -= 1
                else:
                    req_index = i
        # Case 2: Only one request was filled, need to cancel the others
        else:
            for i in range(len(reqs)):
                if reqs[i] not in resourceList:
                    reqs[i].cancel()
                    queue_count[suitable_wards[i].name] -= 1
                else:
                    req_index = i

        # Send patient to the chosen ward.
        env.process(self.enter_service(env, reqs[req_index], suitable_wards[req_index]))
        queue_count[suitable_wards[req_index].name] -= 1

    def enter_service(self, env, resource_req, selected_ward):
        los = self.patient_characteristics[0, 1]
        yield env.timeout(los)
        selected_ward.release(resource_req)
        # TODO: Allow for sequence of resources (i.e. move to next ward).
        #       Update: Have added option to instantiate with known characteristics.
        #       Follow-up stays can be instantiated as new patient with same ID.

# TODO: Add a function to populate wards with existing patients, and generate a LoS for them when sim starts.
# Referral process creates new patients at each time step and assigns them to an available ward.
def daily_referrals(env):
    id_start = 0
    while True:
        # Generate number of new referrals for this time step (time step: 1 day)
        num_referrals = np.random.poisson(global_entry_rate)

        for j in range(num_referrals):
            # Instantiate patient.
            Patient(env, id_start)
            id_start += 1

        # Record occupancies.
        for ward in wards:
            occupancy[ward.name].append((env.now, ward.count))
            queue[ward.name].append((env.now, queue_count[ward.name]))
        # Wait for the next time step
        yield env.timeout(1)


# Create an instance of the event process and start it
env.process(daily_referrals(env))

# Run the simulation
env.run(until=100)

# TODO: Visualise occupancies and queues.
# TODO: Monte-carlo functionality: Export occupancies and queues, calculate ensemble results, add visualisations.
