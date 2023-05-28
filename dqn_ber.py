import gym
import tqdm
import json
import mlflow
import argparse
import datetime
import collections
from os import path
from dynaconf import Dynaconf
from utils import init_ray, check_path, log_with_timeout
from algorithms.ray_dqn_pber import DQNPolicyWithPBER
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from replay_buffer.ber import CustomReplayBuffer

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

# Check path available
log_path = path.join(settings.log.save_file, settings.dqn.env)
check_path(log_path)
log_path = path.join(log_path, "DQN")
check_path(log_path)

# Attributes
batch_size = parser.parse_args().batch_size
sub_buffer_size = parser.parse_args().sub_buffer_size
fragment_length = parser.parse_args().fragment_length
if batch_size != 0:
    settings["ddqn"]["hyper_parameters"]["train_batch_size"] = batch_size
if sub_buffer_size != 0:
    ex_setting["replay_buffer_config"]["sub_buffer_size"] = sub_buffer_size
if fragment_length != 0:
    ex_setting["replay_buffer_config"]["fragment_length"] = fragment_length

# Set mlflow
mlflow.set_tracking_uri(settings.mlflow.url)
mlflow.set_experiment(experiment_name=settings.mlflow.experiment)
mlflow_client = mlflow.tracking.MlflowClient()
mlflow_run = mlflow.start_run(run_name="DQN_PBER_" + datetime.datetime.now().strftime("%Y%m%d"),
                              tags={"mlflow.user": settings.mlflow.user})

# Log parameters
hyper_parameters_log = dict(collections.ChainMap({"type": "PBER"},
                                                 settings.dqn.hyper_parameters.to_dict(),
                                                 settings.dqn.hyper_parameters.to_dict()["replay_buffer_config"],
                                                 ex_setting.replay_buffer_config.to_dict()))
hyper_parameters_log.pop("replay_buffer_config")
mlflow.log_params(hyper_parameters_log)

# Set algorithm
env_example = wrap_deepmind(gym.make(settings.dqn.env))
hyper_parameters = settings.dqn.hyper_parameters.to_dict()
replay_buffer_config = {
    **DQNConfig().replay_buffer_config,
    **settings.dqn.hyper_parameters.replay_buffer_config.to_dict(),
    "type": MultiAgentBatchedPrioritizedReplayBuffer,
    "obs_space": env_example.observation_space,
    "action_space": env_example.action_space,
    "sub_buffer_size": ex_setting.replay_buffer_config.sub_buffer_size,
    "rollout_fragment_length": ex_setting.replay_buffer_config.fragment_length,
}

hyper_parameters["replay_buffer_config"] = replay_buffer_config
algorithm = DQNPolicyWithPBER(hyper_parameters, env=settings.dqn.env)

for i in tqdm.tqdm(range(1, 10000)):
    result = algorithm.train()
    if i == 10 or i % settings.log.log == 0:
        learner_data = result['info']['learner']
        if learner_data.get("time_usage", None) is not None:
            log_with_timeout()
    with open(path.join(log_path, str(i) + ".json"), "w") as f:
        result["config"] = None
        json.dump(result, f)
