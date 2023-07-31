from typing import Dict, List

import pandas as pd
from ml_tools.data_sources.databricks import Databricks
from ml_tools.utils import DataFormat
from pyspark.sql import DataFrame, SparkSession


class ModelUtils:
    """
    Class that contains reusable methods for model classes.
    """

    @staticmethod
    def get_dummy_col_names(data: pd.DataFrame, key: str, dummy_list: Dict) -> List:
        """
        Get dummy columns based on the key in the dummy_list.

        Args:
            data: pd.DataFrame
            key: str Column name of DummyList
            dummy_list: Dict Category of Dummy Controls

        Returns:
            List of dummy cols.

        """

        return [col for col in data if col.startswith(dummy_list[key] + "_")]

    @staticmethod
    def get_empty_df(spark: SparkSession, df_schema) -> DataFrame:
        """

        Creates an empty metrics DataFrame to store results

        Args:
            None

        Returns:
            DataFrame

        """

        empty_rdd = spark.sparkContext.emptyRDD()
        metrics_output_df = spark.createDataFrame(data=empty_rdd, schema=df_schema)
        return metrics_output_df

    @staticmethod
    def check_delta_file_exist(delta_file_path):
        """
        Checks if table exists in spark.
        """
        file_exist = False
        try:
            Databricks().load(delta_file_path, DataFormat.delta)
            file_exist = True
        except:
            pass
        return file_exist