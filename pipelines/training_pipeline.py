from zenml import pipeline
import logging
import sys

sys.path.append("..")
from wind_turbine_scada.steps.ingest_data import ingest_data
from wind_turbine_scada.steps.clean_data import clean_df
from wind_turbine_scada.steps.evaluate_model import evaluate_model
from wind_turbine_scada.steps.train_model import train_model


#
@pipeline()
def train_pipeline(data_path: str):
    logging.info(f"Train pipeline initiated successfully")
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, y_train, X_test, y_test)
    r2_score, rmse = evaluate_model(model, X_test, y_test)
    return r2_score, rmse
