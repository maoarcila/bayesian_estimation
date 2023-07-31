# Databricks notebook source
import os

from beertech_utils.core.config import ConfigManager
from beertech_utils.core.logger import LogManager as LM

from bayesian_estimation.config.config_mappings import ConfigMappings

from bayesian_estimation.config.mms_wamp_partition import (
    MmsWampPartition as ModelConfig,
)
from lola_optimize.elasticity.models.ols.ols_mms_wamp import ols_mms_wamp as OLSModel
from lola_optimize.elasticity.preprocess.preprocess_mms import PreprocessMMS
from lola_optimize.elasticity.preprocess.preprocess_wamp_mms import PreprocessWampMms
from ml_tools.data_sources.databricks import Databricks
from ml_tools.experiment.experiment import Experiment
from ml_tools.experiment.experiment_phases import ExperimentPhases
from ml_tools.experiment.experiment_tags import ExperimentTags
from ml_tools.experiment_phase.experiment_dependencies import ExperimentDependencies
from ml_tools.experiment_phase.experiment_phase_metrics import ExperimentPhaseMetrics
from ml_tools.experiment_phase.experiment_phase_parameters import (
    ExperimentPhaseParameters,
)
from ml_tools.factory import Factory
from ml_tools.utils import ModelFramework, Utils
from nameof import nameof
from pyspark.dbutils import DBUtils

# COMMAND ----------

post_process_seed = os.getenv("post_process_seed", default=None)
Utils.guard_against_none(nameof(post_process_seed), post_process_seed)
print(post_process_seed)
priors_from_ols = True
user_name = DBUtils().notebook.entry_point.getDbutils().notebook().getContext().userName().get()
data_source = Databricks()

experiment_logger = LM.get_logger(__name__)
dd_config = {
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s:%(levelname)s:%(process)d:%(name)s:%(module)s:%(lineno)d:%(message)s",
        "handlers": "DataDogHandler",
        "service": "lola-optimize",
    },
    "datadog": {
        "app_key": Utils.get_secrets(scope="lola-optimize", key="lola-dd-application-key"),
        "api_key": Utils.get_secrets(scope="lola-optimize", key="lola-dd-api-key"),
    },
}
ConfigManager().upsert_config(dd_config)

# COMMAND ----------

config: ModelConfig = Factory.create(ConfigMappings.MmsWampPartition.value)
dependencies: ExperimentDependencies = ExperimentDependencies(
    {nameof(data_source): data_source, nameof(post_process_seed): post_process_seed}
)
experiment_phase_parameters = ExperimentPhaseParameters(
    {
        nameof(config.aggregation_list): config.aggregation_list,
        nameof(config.ab_indicator): config.ab_indicator,
        nameof(config.preprocessed_dataset_path): config.preprocessed_dataset_path,
        nameof(post_process_seed): post_process_seed,
    }
)
experiment_tags = ExperimentTags({"executed_by": user_name})
experiment_name: str = "Integrated_Bayesian_OLS_Experiment_data_creation"
experiment_phases = ExperimentPhases(
    [
        PreprocessMMS.with_model_framework(ModelFramework.none),
        PreprocessWampMms.with_additional_log_parameters(experiment_phase_parameters).with_model_framework(
            ModelFramework.none
        ),
    ]
)
experiment: Experiment = (
    Experiment(experiment_name=experiment_name)
    .with_logger(logger=experiment_logger)
    .with_experiment_phases(phases=experiment_phases)
    .with_config(config=config)
    .with_dependencies(dependencies=dependencies)
    .with_experiment_tags(experiment_tags)
    .run()
)

# COMMAND ----------

if priors_from_ols:
    min_max_list = [(201501, 201912)]
    for min_year_month, max_year_month in min_max_list:
        for mms_name in config.mms_list:
            for wamp_name in config.wamps_list:
                config: ModelConfig = Factory.create(ConfigMappings.MmsWampPartition.value)
                dependencies: ExperimentDependencies = ExperimentDependencies(
                    {
                        nameof(min_year_month): min_year_month,
                        nameof(max_year_month): max_year_month,
                        nameof(wamp_name): wamp_name,
                        nameof(mms_name): mms_name,
                        nameof(post_process_seed): post_process_seed,
                    }
                )
                experiment_phase_parameters = ExperimentPhaseParameters(
                    {
                        nameof(min_year_month): min_year_month,
                        nameof(max_year_month): max_year_month,
                        nameof(config.control_list): config.control_list,
                        nameof(config.price_type): config.price_type,
                        nameof(wamp_name): wamp_name,
                        nameof(mms_name): mms_name,
                        nameof(config.run_id): config.run_id,
                        nameof(post_process_seed): post_process_seed,
                    }
                )
                experiment_phase_metrics = ExperimentPhaseMetrics(
                    {"metric_output_df": None, "final_post_process_output": None}
                )

                experiment_tags = ExperimentTags({"executed_by": user_name, "modelType": "Contraint_OLS"})

                experiment_name: str = "WAMP_MMS_OLS_Experiment"

                experiment_phases = ExperimentPhases(
                    [
                        OLSModel.with_additional_log_metrics(experiment_phase_metrics)
                        .with_additional_log_parameters(experiment_phase_parameters)
                        .with_model_framework(ModelFramework.none),
                    ]
                )
                experiment: Experiment = (
                    Experiment(experiment_name=experiment_name)
                    .with_logger(logger=experiment_logger)
                    .with_experiment_phases(phases=experiment_phases)
                    .with_config(config=config)
                    .with_dependencies(dependencies=dependencies)
                    .with_experiment_tags(experiment_tags)
                    .run()
                )



# Databricks notebook source
import os

from beertech_utils.core.config import ConfigManager
from beertech_utils.core.logger import LogManager as LM

from lola_optimize.elasticity.configs.bayesian_arimax_mms_wamp_config import (
    Bayesian_Arimax_MMS_WAMP_Config,
)
from lola_optimize.elasticity.configs.config_mappings import ConfigMappings
from lola_optimize.elasticity.models.bayesian.bayesian import Bayesian
from ml_tools.data_sources.databricks import Databricks
from ml_tools.experiment.experiment import Experiment
from ml_tools.experiment.experiment_phases import ExperimentPhases
from ml_tools.experiment.experiment_tags import ExperimentTags
from ml_tools.experiment_phase.experiment_dependencies import ExperimentDependencies
from ml_tools.experiment_phase.experiment_phase_metrics import ExperimentPhaseMetrics
from ml_tools.experiment_phase.experiment_phase_parameters import (
    ExperimentPhaseParameters,
)
from ml_tools.factory import Factory
from ml_tools.utils import ModelFramework, Utils
from nameof import nameof
from pyspark.dbutils import DBUtils

# COMMAND ----------
post_process_seed = os.getenv("post_process_seed", default=None)
Utils.guard_against_none(nameof(post_process_seed), post_process_seed)
print(post_process_seed)
mms = os.getenv("mms", default=None)
Utils.guard_against_none(nameof(mms), mms)
print(mms)
wamp = os.getenv("wamp", default=None)
Utils.guard_against_none(nameof(wamp), wamp)
print(wamp)
priors_from_ols = True
user_name = DBUtils().notebook.entry_point.getDbutils().notebook().getContext().userName().get()
data_source = Databricks()

experiment_logger = LM.get_logger(__name__)
dd_config = {
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s:%(levelname)s:%(process)d:%(name)s:%(module)s:%(lineno)d:%(message)s",
        "handlers": "DataDogHandler",
        "service": "lola-optimize",
    },
    "datadog": {
        "app_key": Utils.get_secrets(scope="lola-optimize", key="lola-dd-application-key"),
        "api_key": Utils.get_secrets(scope="lola-optimize", key="lola-dd-api-key"),
    },
}
ConfigManager().upsert_config(dd_config)

# COMMAND ----------

config: Bayesian_Arimax_MMS_WAMP_Config = Factory.create(ConfigMappings.Bayesian_Arimax_MMS_WAMP_Config.value)
train_min_year_month = "201501"
train_max_year_month = "201912"

dependencies: ExperimentDependencies = ExperimentDependencies(
    {
        nameof(train_min_year_month): train_min_year_month,
        nameof(train_max_year_month): train_max_year_month,
        nameof(wamp): wamp,
        nameof(mms): mms,
        nameof(post_process_seed): post_process_seed,
        nameof(priors_from_ols): priors_from_ols,
    }
)
experiment_phase_parameters = ExperimentPhaseParameters(
    {
        nameof(config.control_list): config.control_list,
        nameof(train_min_year_month): train_min_year_month,
        nameof(train_max_year_month): train_max_year_month,
        nameof(wamp): wamp,
        nameof(mms): mms,
        nameof(config.dictionary_of_priors): config.dictionary_of_priors,
        nameof(post_process_seed): post_process_seed,
        nameof(priors_from_ols): priors_from_ols,
    }
)
experiment_phase_metrics = ExperimentPhaseMetrics(
    {"metric_output_df": None, "bayesian_mape": None, "bayesian_wape": None, "final_post_process_output": None}
)
experiment_tags = ExperimentTags({"executed_by": user_name, "modelType": "Bayesian_SARIMAX"})

experiment_name: str = "Bayesian_SARIMAX_Experiment"
experiment_phases = ExperimentPhases(
    [
        Bayesian.with_additional_log_metrics(experiment_phase_metrics)
        .with_additional_log_parameters(experiment_phase_parameters)
        .with_model_framework(ModelFramework.none),
    ]
)
experiment: Experiment = (
    Experiment(experiment_name=experiment_name)
    .with_logger(logger=experiment_logger)
    .with_experiment_phases(phases=experiment_phases)
    .with_config(config=config)
    .with_dependencies(dependencies=dependencies)
    .with_experiment_tags(experiment_tags)
    .run()
)


# Databricks notebook source
import os

from lola_optimize.elasticity.configs.bayesian_arimax_mms_wamp_config import (
    Bayesian_Arimax_MMS_WAMP_Config,
)
from lola_optimize.elasticity.configs.config_mappings import ConfigMappings
from lola_optimize.elasticity.utils.model_utils import ModelUtils
from ml_tools.base import Base
from ml_tools.data_sources.databricks import Databricks
from ml_tools.factory import Factory
from ml_tools.utils import DataFormat

# COMMAND ----------

post_process_seed = os.getenv("post_process_seed", default=None)
config: Bayesian_Arimax_MMS_WAMP_Config = Factory.create(ConfigMappings.Bayesian_Arimax_MMS_WAMP_Config.value)
df = ModelUtils.get_empty_df(Base().spark, config.final_output_schema)
databricks = Databricks()
# COMMAND ----------

for mms in config.mms_list:
    for wamp in config.wamps_list:
        seed_path = f"{config.bayesian_model_output_data_path}{post_process_seed}_{mms}_{wamp}"
        if ModelUtils.check_delta_file_exist(seed_path):
            delta_df = databricks.load(path=seed_path, format=DataFormat.delta)
            df = df.union(delta_df)

display(df)

# COMMAND ----------

df = df.groupBy(["min_year_month", "max_year_month", "mms", "wamp_1"]).pivot("wamp_2").sum("estimation_metric")
databricks.data_frame = df
databricks.write(
    f"{config.bayesian_arimax_mms_wamp_output_path}{post_process_seed}",
    format=DataFormat.delta,
    save_mode=SaveMode.overwrite,
)
display(df)


# Databricks notebook source
import uuid

from beertech_utils.core.config import ConfigManager
from beertech_utils.core.logger import LogManager as LM

from lola_optimize.elasticity.configs.bayesian_arimax_tus import Bayesian_Arimax_TUS

from lola_optimize.elasticity.configs.config_mappings import ConfigMappings
from lola_optimize.elasticity.preprocess.detrend_deseasonalize_phase import (
    DetrendDeseasonalizePhase,
)
from lola_optimize.elasticity.preprocess.preprocess import Preprocess
from ml_tools.data_sources.databricks import Databricks
from ml_tools.experiment.experiment import Experiment
from ml_tools.experiment.experiment_phases import ExperimentPhases
from ml_tools.experiment.experiment_tags import ExperimentTags
from ml_tools.experiment_phase.experiment_dependencies import ExperimentDependencies
from ml_tools.experiment_phase.experiment_phase_parameters import (
    ExperimentPhaseParameters,
)
from ml_tools.factory import Factory
from ml_tools.utils import DataFormat, ModelFramework, Utils
from nameof import nameof
from pyspark.dbutils import DBUtils

# COMMAND ----------

post_process_seed = str(uuid.uuid4().int)
priors_from_ols = True
user_name = DBUtils().notebook.entry_point.getDbutils().notebook().getContext().userName().get()
data_source = Databricks()

experiment_logger = LM.get_logger(__name__)
dd_config = {
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s:%(levelname)s:%(process)d:%(name)s:%(module)s:%(lineno)d:%(message)s",
        "handlers": "DataDogHandler",
        "service": "lola-optimize",
    },
    "datadog": {
        "app_key": Utils.get_secrets(scope="lola-optimize", key="lola-dd-application-key"),
        "api_key": Utils.get_secrets(scope="lola-optimize", key="lola-dd-api-key"),
    },
}
ConfigManager().upsert_config(dd_config)

# COMMAND ----------

config: Bayesian_Arimax_TUS = Factory.create(ConfigMappings.Bayesian_Arimax_TUS.value)
dependencies: ExperimentDependencies = ExperimentDependencies(
    {nameof(data_source): data_source, nameof(post_process_seed): post_process_seed}
)
experiment_phase_parameters = ExperimentPhaseParameters(
    {
        nameof(config.aggregation_list): config.aggregation_list,
        nameof(config.ab_indicator): config.ab_indicator,
        nameof(config.preprocessed_dataset_path): config.preprocessed_dataset_path,
        nameof(post_process_seed): post_process_seed,
    }
)
experiment_tags = ExperimentTags({"executed_by": user_name})
experiment_name: str = "Integrated_Bayesian_OLS_Experiment_data_creation"
experiment_phases = ExperimentPhases(
    [
        Preprocess.with_model_framework(ModelFramework.none),
        DetrendDeseasonalizePhase.with_model_framework(ModelFramework.none),
    ]
)
experiment: Experiment = (
    Experiment(experiment_name=experiment_name)
    .with_logger(logger=experiment_logger)
    .with_experiment_phases(phases=experiment_phases)
    .with_config(config=config)
    .with_dependencies(dependencies=dependencies)
    .with_experiment_tags(experiment_tags)
    .run()
)

# COMMAND ----------

path = "dbfs:/mnt/lola-optimize/sandbox/preprocessed_df_200903749991869905536133747322990790005"
from ml_tools.data_sources.databricks import Databricks
from ml_tools.utils import DataFormat

display(Databricks().load(path=path, format=DataFormat.delta))