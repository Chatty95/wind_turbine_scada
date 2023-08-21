import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union


class DataStrategy(ABC):
    """
    Abstract class for defining strategy for defining data
    """

    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Pre-process data
        """
        pass


class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Pre-process data
        All EDA stuff that has been tried on notebook can be written here
        :param df:
        :return:
        """
        try:
            data = data.drop(["Date/Time"], axis=1)
            data["Wind Speed (m/s)"].fillna(
                data["Wind Speed (m/s)"].median(), inplace=True
            )
            # Drop categorical columns
            data = data.select_dtypes(include=[np.number])
            return data
        except Exception as e:
            logging.info(f"Error in preprocessing data : {e}")
        logging.info(f"Pre-processing data")


class DataSplitStrategy(DataStrategy):
    """
    Strategy for dividing data in training and test set
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(["Theoretical_Power_Curve (KWh)"], axis=1)
            y = data["Theoretical_Power_Curve (KWh)"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=45
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.info(f"Error in splitting data into test and train : {e}")
            raise e


class DataCleaning:
    """
    Class for cleaning data that does:
        1. Pre-processes the data
        2. Divides teh data into train and test
    """

    def __init__(self, data: pd.DataFrame, strategy=DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        :return:

        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.info(f"Error in handling data : {e}")
            raise e
