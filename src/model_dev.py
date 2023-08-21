import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import RegressorMixin


class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """

        :param X_train: training input
        :param y_train: training output
        :return:
        """
        pass

    @abstractmethod
    def optimize(self, **kwargs):
        pass


class LinearREgressionModel(Model):
    """
    Class for implementing Linear Regression
    """

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs
    ) -> RegressorMixin:
        """

        :param X_train:
        :param y_train:
        :return:
        """
        try:
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logging.info(f"Model training Linear Regressor completed")
            return reg
        except Exception as e:
            logging.error(f"Error while training model : {e}")
            raise e

    def optimize(self, **kwargs):
        """
        For linear regression, we are not tuning any hyperparameters
        :param kwargs:
        :return:
        """
        pass


class RandomForestModel(Model):
    """
    Random FOrest Model that implements the Model interface
    """

    def train(self, X_train, y_train, **kwargs):
        try:
            reg = RandomForestRegressor()
            reg.fit(X_train, y_train)
            logging.info(f"Model Randm Forest Regressor training completed")
            return reg
        except Exception as e:
            logging.error(f"Error while training model : {e}")
            raise e

    def optimize(self, **kwargs):
        """
        This function is for hyperparameter tuning
        To be done later
        :return:
        """
        pass
