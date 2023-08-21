from pipelines.training_pipeline import train_pipeline
import logging

if __name__ == "__main__":
    # Run the pipeline
    logging.info(f"Pipeline initiated successfully")
    train_pipeline(
        "/Users/anirban/MLOps_Prod_Grade_Project/wind_turbine_scada/resources/wind_turbine_scada.csv"
    )
