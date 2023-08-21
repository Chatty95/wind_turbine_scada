import logging
import pandas as pd
from zenml import step
from typing import Tuple
import sys

sys.path.append("..")
from wind_turbine_scada.src.data_cleaning import (
    DataCleaning,
    DataSplitStrategy,
    DataPreProcessStrategy,
)


@step()
def clean_df(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,]:
    """
    Cleans the data and divides it into train and test
    :param df: Raw data
    :return: X_train, X_test, y_train, y_test
    """
    try:
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()

        divide_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(df, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info(f"Data cleaned successfully")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.info(f"Error while Data Cleaning : {e}")
        raise e
