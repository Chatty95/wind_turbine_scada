# Predicting power output from turbine based on wind characteristics
- Author : Anirban Chatterjee
- Version : 1.0.0


[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

## Problem Statement

In the context of wind turbines, SCADA systems measure and store data such as wind speed, generated power, and more at 10-minute intervals. This project aims to predict theoretical power values based on features like wind speed and direction. The data is sourced from a wind turbine SCADA system in Turkey.

**Dataset Link:** [Wind Turbine SCADA Dataset](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)

## Content

The dataset includes the following attributes:

- Date/Time (for 10-minute intervals)
- LV ActivePower (kW): Power generated by the turbine
- Wind Speed (m/s): Wind speed at turbine hub height
- Theoretical_Power_Curve (KWh): Theoretical power values based on wind speed
- Wind Direction (°): Wind direction at turbine hub height

The goal is to predict theoretical power output using machine learning techniques. This README explains the project structure, pipeline, and usage of [ZenML](https://zenml.io/) to build, deploy, and manage machine learning workflows.

## Python Requirements

This project requires **Python 3.8**.

(Please make sure Python 3.8 is installed)

For MAC M1 processors, use the following command before creating a virtual environment:
```bash
arch -x86_64 zsh
```

Follow the steps to get started with the project.

```bash
git clone git@github.com:Chatty95/wind_turbine_scada.git
cd wind_turbine_scada
virtualenv wind-scada-3.8 --python=python3.8
source wind-scada-3.8/bin/activate
pip install -r requirements.txt
```

Starting with ZenML 0.20.0, ZenML comes bundled with a React-based dashboard. This dashboard allows you
to observe your stacks, stack components and pipeline DAGs in a dashboard interface. 
To access this, you need to [launch the ZenML Server and Dashboard locally](https://docs.zenml.io/user-guide/starter-guide#explore-the-dashboard)

To launch the ZenML Server and Dashboard:

```bash
zenml up
```

## Run Pipeline

Run the training pipeline using

```commandline
python run_pipeline.py
```
**You shall get a link to the dashboard on your terminal.**

(Looks something like : Dashboard URL: http://127.0.0.1:8237/workspaces/default/pipelines/ecaa96ba-50dd-4ea4-a452-2da5bfa2fdac/runs/0c1da188-119c-48ae-a17e-f96d7f9fa637/dag)**

# Solution Architecture
We're creating an end-to-end pipeline for continuous prediction and deployment. This cloud-deployable pipeline includes:

- Raw data input
- Feature extraction
- Model training and parameters
- Predictions
- Data application

## Deployment

*Deployment is yet to be implemented and I have only worked on creating a ML training pipeline for now.*

If you are running the `run_deployment.py` script, you will also need to install some integrations(MLflow) using ZenML:

```bash
To be done
```

The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component.

```bash
To be done
```


## The Solution

To create a real-world workflow for predicting power output in real-time and improving decision-making, a single model training iteration is insufficient. Instead, we're constructing an end-to-end pipeline that continuously predicts and deploys our machine learning model. This setup includes a data application that leverages the most up-to-date deployed model for business utilization.

Our dynamic pipeline is deployable in the cloud, adaptable to scalability demands, and ensures meticulous tracking of parameters and data across each pipeline run. This comprehensive pipeline encompasses raw data inputs, feature extraction, outcome insights, the machine learning model itself, its parameters, and the resultant predictions. ZenML's capabilities empower us to establish such a pipeline seamlessly, with a balance of simplicity and robustness.

### Training Pipeline

Our standard training pipeline consists of several steps:

- `ingest_data`: This step will ingest the data and create a `DataFrame`.
- `clean_data`: This step will clean the data and remove the unwanted columns. (**All EDA needs to be added here**)
- `train_model`: This step will train the model and save the model using [MLflow autologging](https://www.mlflow.org/docs/latest/tracking.html).
- `evaluate_model`: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.

#  Diving into the code

## Model Training and Optimization 
**(src/model_dev.py)**

The project employs a modular approach to model training and optimization using abstract classes. This ensures flexibility and extensibility in implementing different types of models while maintaining a consistent interface.

### Model Classes

#### Abstract Model (Model)

The `Model` abstract class serves as the foundation for all model implementations. It defines two abstract methods:

- `train(X_train, y_train)`: To train the model using input features and target outputs.
- `optimize(**kwargs)`: To optimize the model's hyperparameters (if applicable).

#### Linear Regression Model (LinearRegressionModel)

The `LinearRegressionModel` class implements linear regression as a model. It inherits from the `Model` class and provides concrete implementations for the abstract methods. Key methods include:

- `train(X_train, y_train, **kwargs)`: Trains a linear regression model on the provided training data. This method returns the trained model instance.
- `optimize(**kwargs)`: No hyperparameter tuning is implemented for linear regression.

#### Random Forest Model (RandomForestModel)

The `RandomForestModel` class implements a random forest regressor. Similar to the `LinearRegressionModel`, it also inherits from the `Model` class and provides concrete implementations for the abstract methods:

- `train(X_train, y_train, **kwargs)`: Trains a random forest regressor model on the given training data. The trained model instance is returned.
- `optimize(**kwargs)`: Hyperparameter tuning for the random forest regressor is not yet implemented.

These classes exemplify how different model types can be easily integrated into the pipeline while adhering to a common interface for training and optimization.

The training process logs relevant information using the `logging` module, ensuring traceability and transparency in the model development process.

## Data Handling Strategies
**(src/data_cleaning.py)**

This project employs a strategy design pattern for data handling, offering a flexible approach to preprocess and split data for machine learning tasks.

### Data Strategy (DataStrategy)

The `DataStrategy` abstract class serves as the base for defining various data handling strategies. It mandates the implementation of the `handle_data(df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]` method, which allows different data-related actions to be encapsulated under different strategies.

### Data Pre-processing Strategy (DataPreProcessStrategy)

The `DataPreProcessStrategy` class focuses on data pre-processing. It inherits from the `DataStrategy` class and offers concrete implementations for data cleaning and feature engineering:

- `handle_data(data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]`: This method performs data pre-processing steps, such as dropping specific columns, imputing missing values, and selecting numerical features. The cleaned data is then returned.

### Data Split Strategy (DataSplitStrategy)

The `DataSplitStrategy` class handles data division into training and testing sets. It also inherits from the `DataStrategy` class and provides implementations for splitting data:

- `handle_data(data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]`: This method splits the input data into features and target variables (X and y). It then further divides the data into training and testing sets using the `train_test_split` function.

### Data Cleaning (DataCleaning)

The `DataCleaning` class encapsulates the entire data cleaning process by coordinating different strategies. It takes input data and a selected strategy upon instantiation:

- `handle_data() -> Union[pd.DataFrame, pd.Series]`: This method applies the chosen data handling strategy to the input data. It orchestrates the sequence of data cleaning steps and returns the pre-processed and split data.

These strategies allow for modularity and customization in data handling processes. The strategies can be easily extended or swapped depending on the specific requirements of the machine learning pipeline.

## Evaluation Strategies
**(src/evaluation.py)**

The project employs an adaptable strategy pattern for evaluating models, allowing for the calculation of various performance metrics.

### Evaluation Strategy (Evaluation)

The `Evaluation` abstract class forms the foundation for defining different evaluation strategies. It mandates the implementation of the `calculate_scores(y_true: np.ndarray, y_pred: np.ndarray)` method, which enables the computation of evaluation metrics based on true and predicted values.

### Mean Squared Error Strategy (MSE)

The `MSE` class exemplifies an evaluation strategy using the Mean Squared Error (MSE) metric. It inherits from the `Evaluation` class and provides a concrete implementation for evaluating model performance using MSE:

- `calculate_scores(y_train: np.ndarray, y_pred: np.ndarray)`: This method computes the MSE score between the true target values and the predicted values. The calculated MSE is returned.

### R-Squared Strategy (R2)

The `R2` class represents an evaluation strategy utilizing the R-Squared (R2) score. Similar to the previous strategies, it inherits from the `Evaluation` class and offers an implementation to evaluate models based on R2:

- `calculate_scores(y_train: np.ndarray, y_pred: np.ndarray)`: This method computes the R2 score, providing insights into the goodness of fit between true and predicted values.

### Root Mean Squared Error Strategy (RMSE)

The `RMSE` class introduces an evaluation strategy employing the Root Mean Squared Error (RMSE) metric. Like other strategies, it inherits from the `Evaluation` class and provides a method to calculate RMSE:

- `calculate_scores(y_train: np.ndarray, y_pred: np.ndarray)`: This method computes the RMSE score, which offers a measure of the average difference between true and predicted values.

These evaluation strategies allow for adaptable assessment of model performance, catering to different business needs and perspectives.


### Deployment Pipeline

**To be done**

While this ZenML Project trains and deploys a model locally, other ZenML integrations such as the [Seldon](https://github.com/zenml-io/zenml/tree/main/examples/seldon_deployment) deployer can also be used in a similar manner to deploy the model in a more production setting (such as on a Kubernetes cluster). 

We shall use MLflow (To be done) here for the convenience of its local deployment.

[//]: # (![training_and_deployment_pipeline]&#40;_assets/training_and_deployment_pipeline_updated.png&#41;)

## Pipelines supported :

You can run two pipelines as follows:

( For now, we only have the training pipeline)

- Training pipeline:

```bash
python run_pipeline.py
```

- The continuous deployment pipeline:

```bash
To be done
```

##  Demo Streamlit App

A live Streamlit demo of this project will be available here, allowing you to input product features and predict customer satisfaction using the latest trained models. 
```bash
To be done
```

# Resources

Here are some of the resources, sites, and notebooks that I referred to during the development of this project:

- [Wind Turbine SCADA Dataset](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset): The dataset used for this project.

- [ZenML Documentation](https://docs.zenml.io/): The official documentation for ZenML, which guided the implementation of the machine learning pipeline.

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html): Documentation for MLflow, which I plan to use for deployment in future iterations.

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html): Documentation for the scikit-learn library, which provided the machine learning models used in this project.

- [Streamlit Documentation](https://docs.streamlit.io/): Documentation for Streamlit, which I used to create the live demo of the project.

- [Python Documentation](https://docs.python.org/3/): The official Python documentation, which was referenced for language syntax and features.

- [Kaggle](https://www.kaggle.com/): Community platform with various datasets, kernels, and discussions that provided insights and inspiration.

- [DataCamp](https://www.datacamp.com/): Online platform for learning data science and machine learning, which helped enhance my understanding of concepts.

- [GitHub](https://github.com/): Version control platform, where I hosted the code and tracked changes throughout the project.

- [Notebook Name](https://colab.research.google.com/github/zenml-io/zenml/blob/main/examples/quickstart/notebooks/quickstart.ipynb): Notebook provided by zenml.io official devs.

I would like to express my gratitude to these resources and communities for their valuable contributions to the development of this project.
