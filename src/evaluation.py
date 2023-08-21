import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    """
    Abstract class for defining evaluation strategy for evaluation of our models
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates teh scores/evaluation metrics for the models
        :param y_true:
        :param y_pred:
        :return:
        """
        pass


class MSE(Evaluation):
    """
    Evaluation strategy using MSE
    """

    def calculate_scores(self, y_train: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_train, y_pred)
            return mse
        except Exception as e:
            logging.error(f"Could not calculate MSE : {e}")
            raise e


class R2(Evaluation):
    """
    Evaluation strategy using R2 score
    """

    def calculate_scores(self, y_train: np.array, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 score")
            r2 = r2_score(y_train, y_pred)
            return r2
        except Exception as e:
            logging.error(f"Could not calculate R2 sccore : {e}")
            raise e


class RMSE(Evaluation):
    """
    Evaluation strategy using MSE
    """

    def calculate_scores(self, y_train: np.array, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            mse = mean_squared_error(y_train, y_pred, squared=False)
            return mse
        except Exception as e:
            logging.error(f"Could not calculate RMSE : {e}")
            raise e
