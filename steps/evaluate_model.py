import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from typing import Tuple
import sys

sys.path.append("..")
from wind_turbine_scada.src.evaluation import MSE, R2, RMSE


@step()
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[float, float]:
    """
    Evaluate model on the ingested data
    :param df:
    :return:
    """

    try:
        prediction = model.predict(X_test)

        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        R2_class = R2()
        r2_score = R2_class.calculate_scores(y_test, prediction)
        print("------------")
        print(r2_score)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        print(rmse)
        print("------------")
        return r2_score, rmse

    except Exception as e:
        logging.error(f"Error in evaluating model : {e}")
        raise e
