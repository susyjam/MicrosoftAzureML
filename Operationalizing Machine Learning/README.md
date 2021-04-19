
[1]: https://github.com/susyjam/MicrosoftAzureML/blob/master/Operationalizing%20Machine%20Learning/images/screen-shot-2020-09-15-at-12.36.11-pm.png
[2]: https://github.com/susyjam/MicrosoftAzureML/blob/master/Operationalizing%20Machine%20Learning/images/Architectural%20Diagram.jpg

# Operationalizing Machine Learning

## Project Overview

In this project, you will continue to work with the Bank Marketing dataset. You will use Azure to configure a cloud-based machine learning production model, deploy it, and consume it. You will also create, publish, and consume a pipeline.

The dataset named 'Bank marketing dataset' contains data about a Telemarketing strategy implemented by a bank. The aim is to predict if a client would subscribe to a term deposit.

## Project main steps
In this project, you will following the below steps:

Authentication
Automated ML Experiment
Deploy the best model
Enable logging
Swagger Documentation
Consume model endpoints
Create and publish a pipeline
Documentation

![Project main steps][1]

## Architectural Diagram

![Architectural Diagram][2]

  1. Register the Dataset: This involves uploading the dataset into Azure Machine Learning Studio. This can be done using the URL or uploading directly from a local folder.
  2. AutoML run: Here we set up configurations like compute cluster, type of machine learning task (in this case classification), exit criterion, etc. This trains different models on our uploaded dataset.
  3. Deploy the best model: Here we select the best performing model from our AutoML run and deploy into production using Azure Container Instance (ACI) or Azure Kubernetes Service (AKS).
  4. Enable logging and Application Insights: This can be done either when deploying the model into production from the studio or afterwards using a python script. This helps us keep track of deployed model performance and number of successful/failed requests.
  5. Consume Model Endpoints: After deploying the model, a REST endpoint is generated and this enables other services to interact with our deployed model. We can send requests to the deployed model and get responses (predictions).
  6. Create and Publish a Pipeline: Using the Azure Python SDK with the aid of a Jupyter Notebook, we can create and publish a pipeline. This requires a config.json file to be present in the working directory. Using pipelines, we can automate the whole process of training and deploying our model.

## Key Steps

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
