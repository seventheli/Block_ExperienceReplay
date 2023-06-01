import gym
import tqdm
import json
import mlflow
import argparse
import datetime
from os import path
from dynaconf import Dynaconf
from ray.rllib.algorithms.dqn import DQN
from algorithms_with_statistics.basic_dqn import DQNWithLogging
from replay_buffer.ber import BlockReplayBuffer
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from utils import init_ray, check_path, logs_with_timeout, convert_np_arrays
from ray.rllib.algorithms.apex_dqn import ApexDQN
from train_with_statistics import train

checkpoint_path = "./checkpoint/"
init_ray("./ray_config.yml")

parser = argparse.ArgumentParser()
parser.add_argument("-S", "--setting", dest="setting_path", type=str)
parser.add_argument("-L", "--with_er_logging", dest="er_logging", type=int, default=0)
parser.add_argument("-SBZ", "--sub_buffer_size", dest="sub_buffer_size", type=int, default=0)

with_er_logging = parser.parse_args().er_logging
sub_buffer_size = parser.parse_args().sub_buffer_size

# Config path
settings = parser.parse_args().setting_path
settings = Dynaconf(envvar_prefix="DYNACONF", settings_files=settings)

# Set MLflow
mlflow.set_tracking_uri(settings.mlflow.url)
mlflow.set_experiment(experiment_name=settings.mlflow.experiment)
mlflow_client = mlflow.tracking.MlflowClient()

# Set hyper parameters
hyper_parameters = settings.apex.hyper_parameters.to_dict()
if sub_buffer_size == 0:
    # Set run object
    run_name = "APEX_DQN_DPER_" + datetime.datetime.now().strftime("%Y%m%d")
    mlflow_run = mlflow.start_run(run_name=run_name,
                                  tags={"mlflow.user": settings.mlflow.user})
    # Log parameters
    mlflow.log_params(hyper_parameters["replay_buffer_config"])
    mlflow.log_params({key: hyper_parameters[key] for key in hyper_parameters.keys() if key not in ["replay_buffer_config"]})

# Set mlflow
mlflow.set_tracking_uri(settings.mlflow.url)
mlflow.set_experiment(experiment_name=settings.mlflow.experiment)
mlflow_client = mlflow.tracking.MlflowClient()
mlflow_run = mlflow.start_run(run_name="APEX_DQN_DPER_" + datetime.datetime.now().strftime("%Y%m%d"),
                              tags={"mlflow.user": settings.mlflow.user})

# Log parameters
hyper_parameters_log = dict(collections.ChainMap({"type": "DPER"},
                                                 settings.apex.hyper_parameters.to_dict(),
                                                 settings.apex.hyper_parameters.to_dict()["replay_buffer_config"]))
hyper_parameters_log.pop("replay_buffer_config")
mlflow.log_params(hyper_parameters_log)

# Set algorithm
hyper_parameters = settings.apex.hyper_parameters.to_dict()
algorithm = ApexDQN(config=hyper_parameters, env=settings.apex.env)

train(algorithm=algorithm,
      settings=settings,
      checkpoint_path=checkpoint_path,
      mlflow_client=mlflow_client,
      mlflow_run=mlflow_run,
      yml_path=setting_path)
