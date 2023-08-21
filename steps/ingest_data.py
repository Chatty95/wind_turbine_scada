import logging
import pandas as pd
from zenml import step


class IngestData:
    """
    Ingest the data from data path
    """

    def __init__(self, data_path: str):
        """
        Instantiate an object of thepip class IngestData
        :param data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingest the data from data path

        :return: pd.DataFrame
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)


@step()
def ingest_data(data_path: str):
    """ "
    Pipeline step to ingest the data from data path

    Args:
        data_path: path to the csv data

    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        logging.info(f"Data ingested successfully")
        return ingest_data.get_data()
    except Exception as e:
        logging.info(f"Error while ingesting the data : {e}")
        raise e
