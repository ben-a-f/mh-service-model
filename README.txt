### Mental Health Service Model ###
This project is a work-in-progress!

The aim is to create an inpatient health service capacity-and-demand forecasting tool that simulates patients with various characteristics occupying beds in inpatient wards.

There are three components to the project:
- Data synthesis: Synthetic patient data will be created with various correlations and characteristics for testing purposes.
- Forecast model: A SimPy discrete simulation model will be developed to handle the forecasting process, including random generation of new referrals according to derived clusters.
- Analysis tools: A suite of tools for outputting simulation results, and a set of visualisations.

Current progress:
- Synthetic data has been created.
- An initial forecast model has been created using SimPy that handles the generation of new patients, assignment to suitable wards, and continuation of existing 'real' patients between the historic and simulation time periods.
- Basic visualisations are outputted.

Next steps:
- The initial model build has a bug causing the derived LoS distributions to be unbounded, leading to unrealistically long-term patients filling the services.
  - Need to investigate the bug, or consider simulating LoS using a regression model rather than a sampling distribution.
