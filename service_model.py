# This script contains the of the SimPy model that describes an inpatient mental health service.

import simpy


def patient(env, name, ward, onboard_time, los):
    # Simulate onboarding/referral process
    yield env.timeout(onboard_time)

    # Request place in ward
    print('%s entering service at %d' % (name, env.now))
    with ward.request() as req:
        yield req

        # Stay in ward
        print('%s starting stay at %s' % (name, env.now))
        yield env.timeout(los)
        print('%s discharged at %s' % (name, env.now))


env = simpy.Environment()
ward = simpy.Resource(env, capacity=2)
for i in range(4):
    env.process(patient(env, 'Patient %d' % i, ward, i+1, 3*(i+1)))
env.run()