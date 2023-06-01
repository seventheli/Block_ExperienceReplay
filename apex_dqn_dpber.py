import gym
import mlflow
import datetime
import argparse
import collections
from dynaconf import Dynaconf
from train_with_statistics import train
from utils import init_ray
from algorithms.ray_apex_dqn_pber import APEXPolicyWithPBER
from ray.rllib.algorithms.apex_dqn.apex_dqn import ApexDQNConfig
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from replay_buffer.pber import MultiAgentBatchedPrioritizedReplayBuffer
checkpoint_path = "./checkpoint/"
init_ray("./ray_config.yml")

parser = argparse.ArgumentParser()
parser.add_argument("-S", "--setting", dest="setting_path", type=str)
parser.add_argument("-EX", "--extra_setting", dest="extra_setting_path", type=str)
parser.add_argument("-BZ", "--batch_size", dest="batch_size", type=int, default=0)
parser.add_argument("-SBZ", "--sub_buffer_size", dest="sub_buffer_size", type=int, default=0)
parser.add_argument("-F", "--fragment_length", dest="fragment_length", type=int, default=0)

# Config path
setting_path = parser.parse_args().setting_path
settings = Dynaconf(envvar_prefix="DYNACONF", settings_files=setting_path)

ber_setting_path = parser.parse_args().extra_setting_path
ex_setting = Dynaconf(envvar_prefix="DYNACONF", settings_files=ber_setting_path)

# Attributes settings
batch_size = parser.parse_args().batch_size
sub_buffer_size = parser.parse_args().sub_buffer_size
fragment_length = parser.parse_args().fragment_length
if batch_size != 0:
    settings["_apex"]["hyper_parameters"]["train_batch_size"] = batch_size
if sub_buffer_size != 0:
    ex_setting["replay_buffer_config"]["sub_buffer_size"] = sub_buffer_size
if fragment_length != 0:
    ex_setting["replay_buffer_config"]["fragment_length"] = fragment_length

# Set mlflow
mlflow.set_tracking_uri(settings.mlflow.url)
mlflow.set_experiment(experiment_name=settings.mlflow.experiment)
mlflow_client = mlflow.tracking.MlflowClient()
mlflow_run = mlflow.start_run(run_name="APEX_DQN_DPBER_" + datetime.datetime.now().strftime("%Y%m%d"),
                              tags={"mlflow.user": settings.mlflow.user})

# Log parameters
hyper_parameters_log = dict(collections.ChainMap({"type": "DPBER"},
                                                 settings.apex.hyper_parameters.to_dict(),
                                                 settings.apex.hyper_parameters.to_dict()["replay_buffer_config"],
                                                 ex_setting.replay_buffer_config.to_dict()))
hyper_parameters_log.pop("replay_buffer_config")
mlflow.log_params(hyper_parameters_log)

# Set algorithm
env_example = wrap_deepmind(gym.make(settings.apex.env))
hyper_parameters = settings.apex.hyper_parameters.to_dict()
replay_buffer_config = {
    **ApexDQNConfig().replay_buffer_config,
    **settings.apex.hyper_parameters.replay_buffer_config.to_dict(),
    "type": MultiAgentBatchedPrioritizedReplayBuffer,
    "obs_space": env_example.observation_space,
    "action_space": env_example.action_space,
    "sub_buffer_size": ex_setting.replay_buffer_config.sub_buffer_size,
    "rollout_fragment_length": ex_setting.replay_buffer_config.fragment_length,
}
hyper_parameters["replay_buffer_config"] = replay_buffer_config



algorithm = APEXPolicyWithPBER(hyper_parameters, env=settings.apex.env)

train(algorithm=algorithm,
      settings=settings,
      checkpoint_path=checkpoint_path,
      mlflow_client=mlflow_client,
      mlflow_run=mlflow_run,
      yml_path=setting_path)
