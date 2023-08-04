from typing import final, Union

import pmdarima as pmd
import pymc3 as pm
import statsmodels.api as sm
import theano.tensor as tt
from ml_tools.data_sources.databricks import Databricks
from ml_tools.experiment_phase.base import Base as ExperimentPhaseBase
from ml_tools.utils import DataFormat, SaveMode, Utils
from nameof import nameof
from pandas import DataFrame as pd_DataFrame
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, concat, lpad

from bayesian_estimation.config.bayesian_arimax_mms_wamp_config import (
    Bayesian_Arimax_MMS_WAMP_Config,
)
from bayesian_estimation.Bayesian.loglike import Loglike
from bayesian_estimation.Sarimax.sarimax import RunSarimaxModel
from bayesian_estimation.model_utils import ModelUtils


class Bayesian(ExperimentPhaseBase):
    """
    This is class for creating Bayesian ARIMAX models. These are the dependencies
    that need to be provided to the experiment_phase to run it successfully:
    1. train_min_year_month str: Minimum year month value to train the arimax model on
    2. train_max_year_month str: Maximum year month value to train the arimax model on
    3. wamp str: WAMP to run the experiment_phase on
    4. mms str: MMS to run the experiment_phase on
    5. post_process_seed str: Post Process seed to make the experiment unique
    6. priors_from_ols: Flag to switch between constraint OLS and ARIMAX values to be used as priors
    """

    databricks: Databricks = None
    config: Bayesian_Arimax_MMS_WAMP_Config
    train_min_year_month: Union[str, None] = None
    train_max_year_month: Union[str, None] = None
    wamp: Union[str, None] = None
    mms: Union[str, None] = None
    post_process_seed: str = ""
    train_data_pandas: pd_DataFrame
    test_data_pandas: pd_DataFrame
    sxmodel: pmd.arima.ARIMA = None
    arimax_mape: Union[float, None] = None
    arimax_wape: Union[float, None] = None
    bayesian_mape: float = None
    bayesian_wape: float = None
    metric_output_df: pd_DataFrame = None
    priors_from_ols: bool = None
    final_post_process_output: DataFrame = None

    def __init__(self):
        ExperimentPhaseBase.__init__(self)
        self.databricks: Databricks = Databricks()
        self.train_min_year_month = self.dependencies.container[nameof(self.train_min_year_month)]
        Utils.guard_against_none(nameof(self.train_min_year_month), self.train_min_year_month)
        self.train_max_year_month = self.dependencies.container[nameof(self.train_max_year_month)]
        Utils.guard_against_none(nameof(self.train_max_year_month), self.train_max_year_month)
        self.wamp: str = self.dependencies.container[nameof(self.wamp)]
        Utils.guard_against_none(nameof(self.wamp), self.wamp)
        self.mms: str = self.dependencies.container[nameof(self.mms)]
        Utils.guard_against_none(nameof(self.mms), self.mms)
        self.priors_from_ols: bool = self.dependencies.container[nameof(self.priors_from_ols)]
        Utils.guard_against_none(nameof(self.priors_from_ols), self.priors_from_ols)
        self.post_process_seed = self.dependencies.container[nameof(self.post_process_seed)]
        Utils.guard_against_invalid_type(nameof(self.post_process_seed), self.post_process_seed, str)

    @final
    def create_train(self) -> None:
        """
        Load the full preprocessed dataset by filtering on min and max year month
        and mms and wamp, and create test and train dataset in pandas for model training.
        Args:
            None
        Returns:
            None
        """
        self.logger.info("Reading Preprocessed data and filtering it over wamp, mms and max and min year month...")
        full_data: DataFrame = self.databricks.load(
            f"{self.config.preprocessed_df_wamp_dataset_path}{self.post_process_seed}", DataFormat.parquet
        ).na.drop()
        full_data = full_data.withColumn(
            self.config.cal_yr_mo_nbr, concat(col(self.config.year), lpad(self.config.month, 2, "0"))
        )
        train_data = full_data.filter(
            (full_data[self.config.cal_yr_mo_nbr] >= self.train_min_year_month)
            & (full_data[self.config.cal_yr_mo_nbr] <= self.train_max_year_month)
        )
        train_data = train_data.filter(
            (train_data[self.config.ab_wamp_value] == self.wamp) & (train_data[self.config.mms] == self.mms)
        ).sort(self.config.cal_yr_mo_nbr)
        test_data = full_data.filter((full_data[self.config.cal_yr_mo_nbr] > self.train_max_year_month))
        self.logger.info("Converting pyspark dataframe to Pandas.")
        self.train_data_pandas = train_data.toPandas()
        self.test_data_pandas = test_data.toPandas()
        if self.priors_from_ols:
            self.ols_model_results_df: DataFrame = self.databricks.load(
                f"{self.config.postprocess_data_path}{self.post_process_seed}", DataFormat.delta
            )

    def train_arimax_model(self) -> None:
        """
        Train Arimax model by adding <<wamp>>_<<mms>>_price_type to the control list and find the best model
        Args:
            None
        Returns:
            None
        """
        self.logger.info("Training Arimax Model and get the best order.")
        self.target_col = f"{self.wamp}_{self.mms}_{self.config.log_vol}"
        log_price_list = [f"{wamp_name}_{self.mms}_{self.config.price_type}" for wamp_name in self.config.wamps_list]
        self.exogenous = self.config.control_list + log_price_list
        self.sxmodel = RunSarimaxModel.train_model(self.train_data_pandas, self.exogenous, self.config, self.target_col)

    def log_arimax_metrics(self) -> None:
        """
        Calculates Arimax model metrics: MAPE and WAPE. If specifies in the experiment notebook, we can log these values in mlflow
        Args:
            None
        Returns:
            None
        """
        pred_obj = self.sxmodel.predict(
            n_periods=len(self.test_data_pandas.index), X=self.test_data_pandas[self.exogenous], return_conf_int=True
        )
        self.test_data_pandas[self.config.forecasted_vol] = pred_obj[0]

        self.test_data_pandas[self.config.ape] = self.test_data_pandas.apply(
            lambda row: self.calculate_ape(row), axis=1
        )
        self.test_data_pandas[self.config.mad] = self.test_data_pandas.apply(
            lambda row: self.calculate_mad(row), axis=1
        )

        self.arimax_mape = float(self.test_data_pandas[self.config.ape].mean())
        self.arimax_wape = float(
            self.test_data_pandas[self.config.mad].sum() / self.test_data_pandas[self.target_col].abs().sum()
        )

    def calculate_ape(self, row):
        """
        Calculate Absolute Percentage Error
        """
        return round(100 * abs((row[self.config.forecasted_vol] / row[self.target_col]) - 1), 3)

    def calculate_mad(self, row):
        """
        Calculate Median absolute deviation
        """
        return round(abs(row[self.config.forecasted_vol] - row[self.target_col]), 3)

    def create_statespace(self) -> None:
        """
        Calculate statespace from the best model order
        Args:
            None
        Returns:
            None
        """
        self.logger.info("Create statespace from the best order...")
        self.mod = sm.tsa.statespace.SARIMAX(
            self.train_data_pandas[self.target_col],
            self.train_data_pandas[self.exogenous],
            order=self.sxmodel.order,
        )
        self.res_mle = self.mod.fit(disp=False)
        self.loglike = Loglike(self.mod)

    def switch_priors(self, x: str):
        """
        Switch priors based on the "priors_from_ols" flag value and
        return from constraint ols model results when True and return from
        ARIMAX statespace params when False.
        Args:
            x str: <<wamp>>_<<mms>>_<<price_type>>
        Returns:
            None
        """
        if self.priors_from_ols:
            prior = self.ols_model_results_df.filter(
                (col(self.config.out_min_year_month) == self.train_min_year_month)
                & (col(self.config.out_max_year_month) == self.train_max_year_month)
                & (col(self.config.out_mms) == self.mms)
                & (col(self.config.out_wamp_1) == self.wamp)
                & (col(self.config.out_wamp_2) == x.split("_")[0])
            ).first()[self.config.out_estimation_metric]
        else:
            prior = self.res_mle.params[x]
        return prior

    def set_priors(self) -> None:
        """
        Set Priors automatically for the Bayesian model.
        If specified in dictionary_of_priors, then take value from the dictionary,
        else, calculate the value from arimax or constraint ols model result.
        Args:
            None
        Returns:
            None
        """
        self.logger.info("Setting all the priors for the model...")
        cross_elasticity_wamp_list = self.config.wamps_list.copy()
        cross_elasticity_wamp_list.remove(self.wamp)
        cross_elasticity_wamp_log_price_list = [
            f"{wamp_name}_{self.mms}_{self.config.price_type}" for wamp_name in cross_elasticity_wamp_list
        ]
        dict_vars = {}
        with pm.Model():
            for x in cross_elasticity_wamp_log_price_list:
                dict_vars[x] = pm.LogNormal(x, self.config.dictionary_of_priors.get(x, self.switch_priors(x)), 1)
            self_elasticity_wamp_log_price = f"{self.wamp}_{self.mms}_{self.config.price_type}"
            dict_vars[self_elasticity_wamp_log_price] = pm.Normal(
                self_elasticity_wamp_log_price,
                self.config.dictionary_of_priors.get(
                    self_elasticity_wamp_log_price, self.switch_priors(self_elasticity_wamp_log_price)
                ),
                0.5,
            )

            if self.sxmodel.order[0] > 0:
                for i in range(self.sxmodel.order[0]):
                    dict_vars[f"{self.config.ar}{i+1}"] = pm.Uniform(f"{self.config.ar}{i+1}", -0.99, 0.99)
            if self.sxmodel.order[2] > 0:
                for i in range(self.sxmodel.order[2]):
                    dict_vars[f"{self.config.ma}{i+1}"] = pm.Uniform(f"{self.config.ma}{i+1}", -0.99, 0.99)

            dict_vars[self.config.sigma] = pm.InverseGamma(self.config.sigma, 2, 4)

            all_vars_list = list(dict_vars.values())
            theta = tt.as_tensor_variable(all_vars_list)

            # use a DensityDist (use a lamdba function to "call" the Op)
            pm.DensityDist("likelihood", self.loglike, observed=theta)
            self.logger.info("Starting Sampling...")
            # Draw samples
            trace = pm.sample(
                self.config.ndraws,
                tune=self.config.nburn,
                return_inferencedata=self.config.return_inferencedata,
                cores=self.config.cores,
                compute_convergence_checks=self.config.compute_convergence_checks,
                chains=self.config.chains,
            )
        self.logger.info("Calculating metrics from the summary...")
        self.summary = pm.summary(trace)

    def log_bayesian_metrics(self) -> None:
        """
        Calculate bayesian model metrics. If specifies in the experiment notebook, we can log these values in mlflow
        Args:
            None
        Returns:
            None
        """
        self.metric_output_df: pd_DataFrame = self.summary[self.config.metrics_to_track]
        params = self.summary["mean"].values
        sarimax_result_wrapper = self.mod.smooth(params)
        prediction_result_wrapper = sarimax_result_wrapper.get_prediction()

        self.train_data_pandas[self.config.forecasted_vol] = prediction_result_wrapper.predicted_mean
        self.train_data_pandas[self.config.ape] = self.train_data_pandas.apply(
            lambda row: self.calculate_ape(row), axis=1
        )
        self.train_data_pandas[self.config.mad] = self.train_data_pandas.apply(
            lambda row: self.calculate_mad(row), axis=1
        )

        self.bayesian_mape = float(self.train_data_pandas[self.config.ape][1:].mean())
        self.bayesian_wape = float(
            self.train_data_pandas[self.config.mad][1:].sum() / self.train_data_pandas[self.target_col][1:].abs().sum()
        )

        output_row = [
            [
                self.train_min_year_month,
                self.train_max_year_month,
                self.mms,
                self.wamp,
                wamp,
                float(
                    self.metric_output_df[self.metric_output_df.index == f"{wamp}_{self.mms}_{self.config.price_type}"][
                        "mean"
                    ].values[0]
                ),
            ]
            for wamp in self.config.wamps_list
        ]
        delta_df = self.spark.createDataFrame(output_row, self.config.final_output_schema)
        self.final_post_process_output = delta_df

    def execute(self) -> "Bayesian":
        self.create_train()
        self.train_arimax_model()
        # self.log_arimax_metrics()
        self.create_statespace()
        try:
            self.set_priors()
            self.log_bayesian_metrics()
        except RuntimeError:
            self.bayesian_mape = 0
            self.bayesian_wape = 0
            self.metric_output_df = 0
            self.final_post_process_output = 0
        return self

    def save(self) -> "Bayesian":
        """
        Save the trained model for predictions and model training features.
        Args:
            None
        Returns:
            None
        """
        if self.final_post_process_output != 0:
            self.logger.info("Saving final result...")
            seed_path = f"{self.config.bayesian_model_output_data_path}{self.post_process_seed}_{self.mms}_{self.wamp}"
            self.databricks.data_frame = self.final_post_process_output
            self.databricks.write(seed_path, format=DataFormat.delta, save_mode=SaveMode.overwrite)
            self.logger.info(f"Saved final result in the delta file at: {seed_path}...")