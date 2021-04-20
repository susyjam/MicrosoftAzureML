# Azure Machine Learning Engineer Capstone Project
# Heart Failure

This project demonstrates how to use an external dataset in a Microsoft Azure Machine Learning workspace; train a model using the different tools available in the AzureML framework as well deploy the model as a web service.

The Azure Auto ML experiment and the Logistic Regression + Hyper Drive experiments were compared and the best performing model generated among the two was chosen and deployed as a webservice REST API using Azure Container Instance (ACI). The REST API endpoint is then consumed using a Python HTTP request calls to produce scoring results.

## Project Workflow diagram

![diagram](https://github.com/susyjam/MicrosoftAzureML/blob/master/Azure%20Machine%20Learning%20Engineer%20Capstone%20Project/images/capstone-diagram.png)

## Dataset

### Overview

The dataset [Heart failure clinical records](https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv).

The dataset contains 299 observations and 12 clinical features with one binary target variable DEATH_EVENT. This data was collected from patients who had heart failures, during their clinical follow-up period.

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.
This information is in [Kaggle site](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data?select=heart_failure_clinical_records_dataset.csv)

![Data1](https://github.com/susyjam/MicrosoftAzureML/blob/master/Azure%20Machine%20Learning%20Engineer%20Capstone%20Project/images/Data1.png)
![Data2](https://github.com/susyjam/MicrosoftAzureML/blob/master/Azure%20Machine%20Learning%20Engineer%20Capstone%20Project/images/Data2.png)
![Data3](https://github.com/susyjam/MicrosoftAzureML/blob/master/Azure%20Machine%20Learning%20Engineer%20Capstone%20Project/images/Data3.png)

The same dataset can also be found on the  and according to the site's description of the data, it states that:
Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

### Task

Our goal is to find the best classification model for predicting death events by heart failure. We will build the models using hyperdrive and automl API from azureml. Given the 12 clinical features in the dataset our model will use DEATH_EVENT as the target column (binary: “1”, means “Yes”, “0” means “No”)

# Automated ML

### Auto ML Settings and Configurations

The AutoML experiment used the following key settings, mostly to limit the compute time and cost.

  * Time limit for experimentation is 60 minutes
  * Iteration should time out in 5 minutes
  * Max cores per iteration set to 2 
  * Max concurrent iterations set to 4
  * Early stopping has been enabled
  * pThe primary metric has been set for accuracy
  * Uses 5 fold cross validation 
  * Limit the number of concurrent iteration to 2 
  * Featurization enabled to allow AutoML to evaludate different feature engineering of the numerical features

```python
# define automl settings
automl_settings = {
    "n_cross_validations": 5,
    "experiment_timeout_minutes": 60,
    "max_concurrent_iterations": 2,
    "primary_metric" : 'accuracy'
}

# define automl configuration settings
automl_config = AutoMLConfig(compute_target=training_cluster,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="DEATH_EVENT",   
                             path = project_folder,
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
```

### Results
The best model produced by the Auto ML run is VotingEnsemble with an accuracy metric of 0.876

#### The screen shot below shows the experiment RunDetails widget of the various models which were created during the automl run. The best model for the experiment VotingEnsemble can be seen at the top of the list.
![run_details1](https://github.com/susyjam/MicrosoftAzureML/blob/master/Azure%20Machine%20Learning%20Engineer%20Capstone%20Project/images/9.png)
![run_details2](images/run_details2.png)

# Hyperparameter Tuning

The pipeline architecture is as follows:

 * Data loaded from TabularDatasetFactory
 * Dataset is further cleaned using a function called clean_data
 * One Hot Encoding has been used for the categorical columns
 * Dataset has been splitted into train and test sets
 * Built a Logistic Regression model
 * Hyperparameters has been tuned using Hyperdrive
 * Selected the best model
 * Saved the model

Benefits of the parameter sampler chosen The project used Random Sampling as it supports early termination of low-performance runs. In random sampling, hyperparameter values are randomly selected from the defined search space with two hyperparameters '--C' (Reqularization Strength) and '--max_iter' (Maximum iterations to converge). Random sampling search is also faster, allows more coverage of the search space and parameter values are chosen from a set of discrete values or a distribution over a continuous range.

The benefits of the early stopping policy chosen The Bandit policy was chosen because it stops a run if the target performance metric underperforms the best run so far by a specified margin. It ensures that we don't keep running the experiment running for too long and end up wasting resources and time looking for the optimal parameter. It is based on slack criteria and a frequency and delay interval for evaluation. Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.

```python
param_sampling = RandomParameterSampling({
    '--C': uniform(0.001, 1.0),
    '--max_iter': choice(0, 10, 50, 100, 150, 200)
})
if "outputs" not in os.listdir():
    os.mkdir("./outputs")

train_script = "./outputs"

shutil.copy('train.py', train_script)


# Create estimator and hyperdrive config
sklearn_env = Environment.get(workspace=ws, name='AzureML-Tutorial')

src = ScriptRunConfig(
    source_directory=train_script,
    script='train.py',
    compute_target=training_cluster,
    environment=sklearn_env
    )

hyperdrive_run_config = HyperDriveConfig(
    run_config=src,
    hyperparameter_sampling=param_sampling,
    policy=early_termination_policy,
    primary_metric_name='Accuracy',
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_total_runs=100,
    max_concurrent_runs=3
)
```

```python
early_termination_policy = BanditPolicy(evaluation_interval=3, slack_factor=0.1, delay_evaluation=3)
```

 1. evaluation_interval: the frequency of applying the policy. An evaluation interval of 3 will apply the policy each time the training script reports the primary metric.
 2. slack_factor: the slack allowed with respect to the best performing training run. Supposed the best performing run at interval 3 with a reported primary metric of 0.85 with a goal to maximize the primary metric.
 3. delay_evaluation: delays the first policy evaluation for a specified number of intervals.

### Hyperdrive Results

The best model from the Hyperdrive + Logistic Regression run produced an accuracy of 0.813

### Model Improvement

To improve the model:

 * Bayesian sampling would also be leverage as it uses trials from previous runs as a prior knowledge to pick new samples and to improve the primary metric.
 * Grid sampling may provide a little bit of performance as it searches over all possible values.
 * Increasing the max_total_runs value can also provide quite significant performance.

The screenshots below shows the RunDetails widget of the hyperdrive run and the best model trained with its parameters.
![hp3](https://github.com/susyjam/MicrosoftAzureML/blob/master/Azure%20Machine%20Learning%20Engineer%20Capstone%20Project/images/3.png)
![hp7](https://github.com/susyjam/MicrosoftAzureML/blob/master/Azure%20Machine%20Learning%20Engineer%20Capstone%20Project/images/7.png)

#### The best Model

![hp5](https://github.com/susyjam/MicrosoftAzureML/blob/master/Azure%20Machine%20Learning%20Engineer%20Capstone%20Project/images/4.png)
![hp5](https://github.com/susyjam/MicrosoftAzureML/blob/master/Azure%20Machine%20Learning%20Engineer%20Capstone%20Project/images/5.png)

# Model Deployment

The HyperDrive model resulted in an accuracy of 0.831 which was slightly lower than the AutoML accuracy.
Below is how the model was deployed:

1. The best model model.pkl file from the Auto ML run was first retrieved together with its conda environment script conda_env.yml and scoring script score.py.
2. An inference configuration was created from the downloaded conda_env.yml and score.py to make sure that the software dependencies and resources needed for deployment is intact.
3. The model was then deployed with the inference configuration as an Azure Container Instance (ACI) webservice.
4. The model was then registered as a model in the Azure ML workspace.

Compared to HyperDrive, AutoML architecture is quite superior, which enables to training 'n' number of models efficiently.

The reason in accuracies might be due to the fact that that we used less number of iterations in AutoML run, which might give better results with more iterations. AutoML also provides a wide variety of models and preprocessing steps which are not carried out Hyperdrive. However, the difference was quite small.

The AutoML model was then deployed as follows:
 * Get the best model from the training run
 * Register the model
 * Download the score python file from azure ml 
 * Create an inference configuration
 * Create an ACI deployment configuration
 * Deploy the model
 * Enable Application Insights

To consume the model
 * We copy the rest endpoint
 * Create a sample JSON payload and post the payload to the endpoint using a HTTP Post
 * Running endpoint


### Model endpoint

The screenshot below shows the healthy status of the deployed model endpoint which indicates the model deployed is active.

![endpoint](images/deployed_endpoint.png)

### Consume deployed model with sample

1. To consume the deployed model the http endpoint url was embeded into a python code with sample data observations loaded into a json payloader.
1. HTTP webservice POST request is then sent to the enpoint url and the response from the url is captured as a json data.
1. Any application or service is capable of consuming the model using the enpoint url by making http request calls to the webservice.

## Screen Recording

The link below provides access to the video demonstration of a working deployed model.

Apparently my microphone failed, sorry for the inconvenience, so I tried to detail everything. I would like to fix it again but I only have access to the course for a one hour, could you excuse my audio?

[![Video](https://github.com/susyjam/MicrosoftAzureML/blob/master/Azure%20Machine%20Learning%20Engineer%20Capstone%20Project/video/Capstone%20-%20Azure%20Machine%20Learning%20Engineer.mp4)](https://www.youtube.com/watch?v=jlYA27ieUg0)

## SRT Video

This screencast is part of my final captain project for the microsoft sponsored udacity program machine learning engineer with microsoft azure in the screencast to experiment where run the first experiment was run as an aja automated machine learning experiment and the second experiment was run as a hyper parameter tuning using hyper drive and logistic regression now the automated ml experiment has this notebook all the details of how the experiment was run and configured can be found in my readme page below on github so over here i first of all imported the libraries initialize the workspace and then here I prepared the data sets and then created a compute cluster then over here i defined the automl configuration i submitted the run onto the remote compute cluster and then autumnal generated several models over here we can see voting ensemble with 87.63  is the best model which was chosen here so in going through these are all the run details that was produced by auto ml and then the next thing the best model was actually retrieve  and then details of the best model can be seen here so after retrieving the best model the next thing is to deploy but before I deploy the best model I came to the hyper parameter tuning with hyperdrive and then did the same process as I did initially for the o2ml and then hyperdrive i configured a hyperdrive configuration settings here submitted my run onto the compute cluster and here hyperdrive produced several models here that we can see most of them are having the same similar performance so one of the best performers was chosen which let's say the first one and then the best run that we can see is 81 uh percent 81.33 which makes auto ml run very much uh uh the best model for deployment so automo run the best model produced by o2ml was deployed and this is the script which was used in deploying the model so over here if we go back here we can check the model which was deployed and here is the deployment which we actually deployed as an azure container instance we can see the rest endpoint of the model and then we can see the deployed state as healthy which shows the active model which has been deployed is very healthy for consumption so over here to consume the model here is my data point that i loaded onto the json payload and then when i have when i execute this python script is going to make a an http post call to this url here so let's see when i make um a request it goes and then it scores whatever thing uh whatever results that is produced by the model so over here we can see this is the result which was produced by the model mean my model is actually producing their score so this is a very efficient and this is a working model which we can actually see so here you can see the results which was produced we seek to predict um um heart failure uh mortality caused by heart failure so one here meaning that there is a probability of immortality which can be caused by heart failure and then zero means there is actually less probability of uh mortality to be caused by heart failure so if you want details of this run you can find it in my github link below thank you
