import pandas as pd

class sarimax_config(Config):

    def __init__(self) -> None:

        self.start_p = 3,
        self.start_q = 3,
        self.adf = "adf"
        self.maximum_p = 5,
        self.maximum_q = 5,  # maximum p and q
        self.series_frequency = 12,  # frequency of series, 12=monthly
        self.differencing_operations = None,  # 0 if stationary, None -> let model determine 'd',which will be slow
        self.seasonal = 1,
        self.seasonal_p = 12,  # Seasonal P
        self.seasonal_q = 12,  # Seasonal Q
        self.seasonal_d = 1,  # Seasonal D
        self.trace = True,
        self.error_action = None,
        self.suppress_warnings = True,
        self.stepwise = True,
        self.random_state = True,