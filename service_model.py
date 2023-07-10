# This WIP script contains the SimPy model that describes an inpatient mental health service.
# There is a "Set manual parameters." section that needs to be completed.

import simpy
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from get_model_parameters import *
import matplotlib.pyplot as plt

# Import data, format and scale.
synthetic_patients = pd.read_csv("synthetic_patients.csv")
transform_cols = ["Age", "LoS", "DailyContacts"]
scaler = RobustScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(synthetic_patients.dropna()[transform_cols]))
scaled_data.columns = transform_cols

# Get model parameters
for col in ["EntryDate", "DischargeDate"]:
    synthetic_patients[col] = pd.to_datetime(synthetic_patients[col], format="%Y-%m-%d")
historic_end_date = max(synthetic_patients["DischargeDate"].max(), synthetic_patients["EntryDate"].max())
global_props, global_centres, global_covs = get_cluster_parameters(scaled_data, 3)
global_entry_rate = get_referral_rate(synthetic_patients["EntryDate"])

# Create the simulation environment
env = simpy.Environment()
ward1 = simpy.Resource(env, capacity=25)
ward2 = simpy.Resource(env, capacity=10)
ward3 = simpy.Resource(env, capacity=10)
ward1.name = "General Ward"
ward2.name = "Senior Ward"
ward3.name = "Long Stay"
# This vector indirectly acts as a priority.
wards = [ward3, ward2, ward1]

# Get historical data
historical_occupancy = get_historical_occupancy(synthetic_patients, wards)

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
    def __init__(self, env, id_num, known_characteristics=None, known_ward=None):
        self.env = env
        self.id = id_num
        self.cluster = np.random.choice(range(len(global_props)), 1, p=global_props)[0]
        # Allow existing patients to keep known characteristics.
        if known_characteristics is None:
            self.patient_characteristics = truncated_mv_normal(scaler,
                                                               global_centres[self.cluster],
                                                               global_covs[self.cluster],
                                                               global_scaled_bounds)
        else:
            # Generate characteristics in line with what is known, and keep the known age.
            new_characteristics = truncated_mv_normal(scaler,
                                                      global_centres[self.cluster],
                                                      global_covs[self.cluster],
                                                      known_characteristics)
            new_characteristics[0][0] = scaler.inverse_transform(known_characteristics.reshape(1, -1))[0][0]
            self.patient_characteristics = new_characteristics
        self.action = env.process(self.choose_service(env, known_ward))

    # Find possible services for this patient and accept first available space.
    def choose_service(self, env, known_ward=None):
        if known_ward is None:
            suitable_wards = []
            for ward in wards:
                if np.all(np.greater_equal(self.patient_characteristics, global_ward_reqs_low[ward.name])) and np.all(
                        np.less_equal(self.patient_characteristics, global_ward_reqs_high[ward.name])):
                    suitable_wards.append(ward)
        else:
            suitable_wards = [known_ward]

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


# Referral process creates new patients at each time step and assigns them to an available ward.
def daily_referrals(env, current_patients=None):

    # Load in existing patients.
    id_start = 0
    # Calculate their LoS to-date. This will be a lower bound for their total LoS.
    current_patients["LoS"] = (historic_end_date - current_patients["EntryDate"]).dt.days
    for idx, row in current_patients.iterrows():
        # Get and scale characteristics: Age, LoS, DailyContacts
        known_characteristics = scaler.transform(np.array(row.values[0:3]).reshape(1, -1))
        # Replace any NAs with global lower bounds.
        known_characteristics[np.isnan(known_characteristics)] = global_scaled_bounds[np.isnan(known_characteristics)]
        # Get Ward object.
        known_ward = next((ward for ward in wards if ward.name == row.values[5]), None)
        # Instantiate patients with known characteristics.
        Patient(env, id_start, known_characteristics, known_ward)
        id_start += 1

    # Start forward sim.
    while True:
        # Generate number of new referrals for this time step (time step: 1 day)
        num_referrals = np.random.poisson(global_entry_rate)

        for j in range(num_referrals):
            # Instantiate patient.
            Patient(env, id_start)
            id_start += 1

        # Record occupancies.
        if env.now != 0:
            for ward in wards:
                occupancy[ward.name].append((env.now, ward.count))
                queue[ward.name].append((env.now, queue_count[ward.name]))
        # Wait for the next time step
        yield env.timeout(1)


# Get current patients.
current_patients = synthetic_patients.loc[synthetic_patients["DischargeDate"].isnull()].copy()
# Create an instance of the event process and start it
env.process(daily_referrals(env, current_patients))

# Run the simulation
env.run(until=1000)

# Convert model occupancy to dataframe in same format as historical occupancy.
occupancy_dfs = {key: pd.DataFrame(value, columns=['Day', 'Occupancy']) for key, value in occupancy.items()}
queue_dfs = {key: pd.DataFrame(value, columns=['Day', 'Queue']) for key, value in queue.items()}

for key in occupancy_dfs:
    occupancy_dfs[key]["Day"] = max(historical_occupancy["Long Stay"].index) + pd.to_timedelta(occupancy_dfs[key]["Day"], unit="D")
    occupancy_dfs[key].set_index("Day", inplace=True)

    queue_dfs[key]["Day"] = max(historical_occupancy["Long Stay"].index) + pd.to_timedelta(queue_dfs[key]["Day"], unit="D")
    queue_dfs[key].set_index("Day", inplace=True)

# Occupancy plots.
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
for i, key in enumerate(historical_occupancy.keys()):
    historic_df = historical_occupancy[key]
    forecast_df = occupancy_dfs[key]
    queue_df = queue_dfs[key]
    ax = axs[i]

    ax.plot(historic_df.index, historic_df['Occupancy'], color='blue', label='Historic')
    ax.plot(forecast_df.index, forecast_df['Occupancy'], color='red', label='Forecast')
    ax.plot(queue_df.index, queue_df['Queue'], color='green', label='Forecast')

    ax.set_title(key)
    ax.legend()

plt.tight_layout()
plt.show()

# TODO: Monte-carlo functionality: Export occupancies and queues, calculate ensemble results, add visualisations.

# TODO: Model is over-estimating either entry rate, LoS, or both. Test, and maybe consider regression-based model rather than cluster-based.
