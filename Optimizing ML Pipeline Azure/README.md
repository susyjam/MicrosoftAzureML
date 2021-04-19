# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data pertaining to direct marketing campaigns of a banking institution. The marketing campaigns were based on phone calls to convince potenitial clients to subscribe to bank's term deposit. We seek to predict whether the potential client would accept and make a term deposit at the bank or not.


The best performing model found using AutoML was a Voting Ensemble with 91.6% accuracy, while the accuarcy of Logistic classifier implemented using hyperdrive was 90.7%
