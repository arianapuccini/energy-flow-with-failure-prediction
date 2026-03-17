# energy-flow-with-failure-prediction
# Vehicle Fault Detection — ML + Compartmental Modeling

A pipeline that combines a Python machine learning classifier with a Julia ODE model to simulate, detect, and analyze vehicle subsystem failures.

This project approaches vehicle fault detection from two angles; Python trains a logistic regression classifier on simulated sensor data to predict engine failures from simulated real-time data. Julia models the underlying physics of energy flow between vehicle subsystems using an ODE generated from a linear compartmental model, and applies the Python-trained model coefficients directly to evaluate fault risk is the highest. The two components are linked by exported CSV files that the Python script produces.

truckPredictor.py
1. Simulates 5,000 sensor readings across engine temperature, vibration, oil pressure, and RPM
2. Assigns probabilistic failure labels based on a risk score; high temperature, high vibration, and low oil pressure each contribute risk without being the sole cause of failure
3. Trains a logistic regression classifier with balanced class weights to handle the minority failure class
4. Evaluates the model using 5 fold cross validation with a pipeline to prevent data leakage
5. Exports sensor data and standardized model coefficients as CSV files

heatFlow.jl
1. Creates a linear compartmental model with four vertices showing energy flow between engine, cooling system, battery, and transmission:
         |transmission| < -------- ||engine|| ------- > |cooling| -------> |battery|
3. Solves the ODE over 30 seconds using the Tsit5 solver
4. Performs eigenvalue analysis to extract the system's time constants
5. Loads the Python-generated sensor data and compares failure rates across all samples vs samples where engine heat is above 95 degrees C
6. Reimplements the logistic regression using the exported coefficients to estimate fault probability at a given operating point
7. Produces combined plots overlaying the ODE energy curve with the ML risk window

All sensor data is simulated; this project is only intended as a modeling and methods demonstration
