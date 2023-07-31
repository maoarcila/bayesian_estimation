from typing import List, Union

import pmdarima as pm
from ml_tools.data_sources.databricks import Databricks
from ml_tools.experiment_phase.base import Base as ExperimentPhaseBase
from ml_tools.utils import DataFormat, Utils
from nameof import nameof
from pandas import DataFrame as pd_DataFrame
from pyspark.sql import DataFrame

from bayesian_estimation.sarimax_config import sarimax_config
from bayesian_estimation.sarimax import RunSarimaxModel


class SarimaxBase(ExperimentPhaseBase):
    """
    This is the base class to create SARIMAX models.
    """

    databricks: Databricks = None
    config: sarimax_config
    exogenous: Union[List, None] = None
    min_year_month: Union[int, None] = None
    max_year_month: Union[int, None] = None
    target_col = None
    test_month_split: int = 3
    post_process_seed: str = ""
    train_data_pandas: pd_DataFrame
    test_data_pandas: pd_DataFrame
    sxmodel: pm.arima.ARIMA = None
    all_features: List[str]
    pvalue_metric: Union[float, None] = None
    estimation_metric: Union[float, None] = None
    mape: Union[float, None] = None
    wape: Union[float, None] = None

    def __init__(self):
        ExperimentPhaseBase.__init__(self)
        self.databricks: Databricks = Databricks()
        self.exogenous = self.dependencies.container[nameof(self.exogenous)]
        Utils.guard_against_empty_list(nameof(self.exogenous), self.exogenous)
        self.min_year_month = self.dependencies.container[nameof(self.min_year_month)]
        Utils.guard_against_none(nameof(self.min_year_month), self.min_year_month)
        self.max_year_month = self.dependencies.container[nameof(self.max_year_month)]
        Utils.guard_against_none(nameof(self.max_year_month), self.max_year_month)
        self.target_col = self.dependencies.container[nameof(self.target_col)]
        Utils.guard_against_none(nameof(self.target_col), self.target_col)
        self.test_month_split = self.dependencies.container[nameof(self.test_month_split)]
        self.post_process_seed = self.dependencies.container[nameof(self.post_process_seed)]
        Utils.guard_against_invalid_type(nameof(self.post_process_seed), self.post_process_seed, str)

    def create_train_test(self) -> None:
        """
        Load the full preprocessed dataset and create train and test dataset in pandas for model training.
        Args:
            None
        Returns:
            None
        """
        full_data: DataFrame = self.databricks.load(
            f"{self.config.preprocessed_dataset_path}{self.post_process_seed}", DataFormat.delta
        ).na.drop()
        filtered_data = full_data.filter(
            f"{self.config.cal_yr_mo_nbr} >= {self.min_year_month} and {self.config.cal_yr_mo_nbr} <= {self.max_year_month}"
        )

        max_year_month = int(filtered_data.select(self.config.cal_yr_mo_nbr).rdd.max()[0])
        train_data = filtered_data.filter(
            f"{self.config.cal_yr_mo_nbr} < { max_year_month - self.test_month_split - 1}"
        ).sort(self.config.cal_yr_mo_nbr)
        test_data = filtered_data.filter(
            f"{self.config.cal_yr_mo_nbr} >= { max_year_month - self.test_month_split - 1}"
        ).sort(self.config.cal_yr_mo_nbr)

        self.train_data_pandas = train_data.toPandas()
        self.test_data_pandas = test_data.toPandas()

    def filter_on_mms(self) -> None:
        # No need to filter on mms in TUS version
        pass

    def filter_on_wamp(self) -> None:
        # No need to filter on wamp in TUS version
        pass

    def train_model(self) -> None:
        self.sxmodel = RunSarimaxModel.train_model(self.train_data_pandas, self.exogenous, self.config, self.target_col)

    def log_metrics(self) -> None:
        result_attribute = getattr(self.sxmodel, "arima_res_")
        self.pvalue_metric = float(getattr(result_attribute, "pvalues")[self.config.log_real_price])
        self.estimation_metric = float(getattr(result_attribute, "params")[self.config.log_real_price])

        pred_obj = self.sxmodel.predict(
            n_periods=len(self.test_data_pandas.index), X=self.test_data_pandas[self.exogenous], return_conf_int=True
        )
        self.test_data_pandas["forecasted_vol"] = pred_obj[0]

        self.test_data_pandas["ape"] = self.test_data_pandas.apply(lambda row: self.calculate_ape(row), axis=1)
        self.test_data_pandas["mad"] = self.test_data_pandas.apply(lambda row: self.calculate_mad(row), axis=1)

        self.mape = float(self.test_data_pandas["ape"].mean())
        self.wape = float(self.test_data_pandas["mad"].sum() / self.test_data_pandas[self.config.log_vol].abs().sum())

    def log_summary(self) -> None:
        """
        Logs summary of the sarimax model.
        """
        summary = self.sxmodel.summary()
        self.logger.info(summary)

    def calculate_ape(self, row):
        """
        Calculate Absolute Percentage Error
        """
        return round(100 * abs((row["forecasted_vol"] / row[self.config.log_vol]) - 1), 3)

    def calculate_mad(self, row):
        """
        Calculate Median absolute deviation
        """
        return round(abs(row["forecasted_vol"] - row[self.config.log_vol]), 3)

    def execute(self) -> "SarimaxBase":
        self.create_train_test()
        self.filter_on_mms()
        self.filter_on_wamp()
        self.train_model()
        self.log_metrics()
        self.log_summary()
        return self

    def save(self) -> "SarimaxBase":
        """
        Save the trained model for predictions and model training features.
        Args:
            None
        Returns:
            None
        """
        pass