import logging
import pandas as pd
from zenml import step
import sys

sys.path.append("..")
from wind_turbine_scada.src.model_dev import LinearREgressionModel
from wind_turbine_scada.src.model_dev import RandomForestModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig


@step()
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> RegressorMixin:
    """

    :param X_train:
    :param y_train:
    :param Y_train:
    :param y_test:
    :param config
    :return: RegressionMix
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearREgressionModel()
            trained_model = model.train(X_train, y_train)
            logging.info(f" {config.model_name} Model training completed")
            return trained_model
        elif config.model_name == "RandomForestRegression":
            model = RandomForestModel()
            trained_model = model.train(X_train, y_train)
            logging.info(f" {config.model_name} Model training completed")
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        logging.error("Error while training model {e}")
        raise e
