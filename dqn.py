import mlflow
import argparse
import datetime
import collections
from dynaconf import Dynaconf
from utils import init_ray
from ray.rllib.algorithms.dqn import DQN
from train_with_statistics import train

checkpoint_path = "./checkpoint/"
init_ray("./ray_config.yml")

parser = argparse.ArgumentParser()
parser.add_argument("-S", "--setting", dest="setting_path", type=str)
parser.add_argument("-BZ", "--batch_size", dest="batch_size", type=int, default=0)

# Config path
setting_path = parser.parse_args().setting_path
settings = Dynaconf(envvar_prefix="DYNACONF", settings_files=setting_path)

# Attributes
batch_size = parser.parse_args().batch_size
if batch_size != 0:
    settings["dqn"]["hyper_parameters"]["train_batch_size"] = batch_size

# Set mlflow
mlflow.set_tracking_uri(settings.mlflow.url)
mlflow.set_experiment(experiment_name=settings.mlflow.experiment)
mlflow_client = mlflow.tracking.MlflowClient()
mlflow_run = mlflow.start_run(run_name="DQN_PER_" + datetime.datetime.now().strftime("%Y%m%d"),
                              tags={"mlflow.user": settings.mlflow.user})

# Log parameters
hyper_parameters_log = dict(collections.ChainMap({"type": "PER"},
                                                 settings.dqn.hyper_parameters.to_dict(),
                                                 settings.dqn.hyper_parameters.to_dict()["replay_buffer_config"]))
hyper_parameters_log.pop("replay_buffer_config")
mlflow.log_params(hyper_parameters_log)

# Set algorithm
hyper_parameters = settings.dqn.hyper_parameters.to_dict()
algorithm = DQN(config=hyper_parameters, env=settings.dqn.env)

train(algorithm=algorithm,
      settings=settings,
      checkpoint_path=checkpoint_path,
      mlflow_client=mlflow_client,
      mlflow_run=mlflow_run,
      yml_path=setting_path)
