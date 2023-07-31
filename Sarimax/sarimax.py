from lola_optimize.elasticity.models.sarimax.base import SarimaxBase
from lola_optimize.elasticity.models.sarimax.sarimax_mms import SarimaxMMS

from typing import List

import pmdarima as pm
from nameof import nameof
from pandas import DataFrame as pd_DataFrame

from lola_optimize.elasticity.configs.sarimax_config import sarimax_config


class RunSarimaxModel:
    @staticmethod
    def train_model(train_data_scaled: pd_DataFrame, exogenous, config: sarimax_config, target_column) -> pm.ARIMA:
        """
        Used to train the model. Where "test" is the Augmented Dickey Fuller test (ADF Test) is a common statistical test used to test whether a given Time series is stationary or not. It is one of the most commonly used statistical test when it comes to analyzing the stationary of a series.
        Args:
            train_data_scaled: The scaled training data.
            exogenous: The exogenous data.
            config: The configuration object.
        Returns:
            The trained model.
        """

        sxmodel = pm.auto_arima(
            train_data_scaled[[target_column]],
            X=train_data_scaled[exogenous],
            start_p=config.start_p,
            start_q=config.start_q,
            test=nameof(config.adf),  # use adftest to find optimal 'd' if not stationary
            max_p=config.maximum_p,
            max_q=config.maximum_q,  # maximum p and q
            m=config.series_frequency,  # frequency of series, 12=monthly
            d=config.differencing_operations,  # 0 if stationary, None -> let model determine 'd',which will be slow
            seasonal=config.seasonal,
            start_P=config.seasonal_p,  # Seasonal P
            start_Q=config.seasonal_q,  # Seasonal Q
            D=config.seasonal_d,  # Seasonal D
            trace=config.trace,
            error_action=config.error_action,
            suppress_warnings=config.suppress_warnings,
            stepwise=config.stepwise,
            random_state=config.random_state,
        )
        return sxmodel
