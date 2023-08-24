import argparse
import datetime
import gym
import json
import mlflow
import os
import pickle
import ray
import torch
import tqdm
from algorithms.apex_ddqn_pber import ApexDDQNWithDPBER
from dynaconf import Dynaconf
from func_timeout import FunctionTimedOut
from mlflow.exceptions import MlflowException
from os import path
from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from ray.tune.logger import UnifiedLogger
from replay_buffer.mpber import MultiAgentPrioritizedBlockReplayBuffer
from utils import init_ray, check_path, logs_with_timeout, convert_np_arrays

torch.manual_seed(10)
parser = argparse.ArgumentParser()
parser.add_argument("-R", "--run_name", dest="run_name", type=int)
parser.add_argument("-S", "--setting", dest="setting_path", type=str)
parser.add_argument("-L", "--log_path", dest="log_path", type=str)
parser.add_argument("-C", "--checkpoint_path", dest="checkpoint_path", type=str)
parser.add_argument("-SBZ", "--sub_buffer_size", dest="sub_buffer_size", type=int, default=0)
parser.add_argument("-R", "--ray", dest="single_ray", type=int, default=0)

if parser.parse_args().single_ray == 1:
    init_ray()
else:
    init_ray("./ray_config.yml")

sub_buffer_size = parser.parse_args().sub_buffer_size

# Config path
log_path = parser.parse_args().log_path
checkpoint_path = parser.parse_args().checkpoint_path
settings = parser.parse_args().setting_path
settings = Dynaconf(envvar_prefix="DYNACONF", settings_files=settings)

# Set MLflow
mlflow.set_tracking_uri(settings.mlflow.url)
mlflow.set_experiment(experiment_name=settings.mlflow.experiment)
mlflow_client = mlflow.tracking.MlflowClient()

# Set hyper parameters
hyper_parameters = settings.apex.hyper_parameters.to_dict()
hyper_parameters["logger_config"] = {"type": UnifiedLogger, "logdir": checkpoint_path}
print("log path: %s \n check_path: %s" % (log_path, checkpoint_path))
if hyper_parameters["double_q"]:
    run_name = "APEX_DDQN"
else:
    run_name = "APEX_DQN"

if sub_buffer_size == 0:
    # Set run object
    run_name = settings.apex.env + "_DPER_%d" % parser.parse_args().run_name
    mlflow_run = mlflow.start_run(run_name=run_name,
                                  tags={"mlflow.user": settings.mlflow.user})
    # Log parameters
    mlflow.log_params(hyper_parameters["replay_buffer_config"])
    mlflow.log_params(
        {key: hyper_parameters[key] for key in hyper_parameters.keys() if key not in ["replay_buffer_config"]})
    algorithm = ApexDQN(config=hyper_parameters, env=settings.apex.env)
else:
    # Set run object
    run_name = settings.apex.env + "_DPBER_%d" % parser.parse_args().run_name
    mlflow_run = mlflow.start_run(run_name=run_name,
                                  tags={"mlflow.user": settings.mlflow.user})
    env_example = wrap_deepmind(gym.make(settings.apex.env))
    # Log parameters
    mlflow.log_params({
        **settings.apex.hyper_parameters.replay_buffer_config.to_dict(),
        "type": "MultiAgentPrioritizedBlockReplayBuffer",
        "sub_buffer_size": sub_buffer_size,
    })
    mlflow.log_params(
        {key: hyper_parameters[key] for key in hyper_parameters.keys() if key not in ["replay_buffer_config"]})
    # Set BER
    replay_buffer_config = {
        **settings.apex.hyper_parameters.replay_buffer_config.to_dict(),
        "type": MultiAgentPrioritizedBlockReplayBuffer,
        "capacity": int(settings.apex.hyper_parameters.replay_buffer_config.capacity),
        "obs_space": env_example.observation_space,
        "action_space": env_example.action_space,
        "sub_buffer_size": sub_buffer_size,
        "worker_side_prioritization": False,
        "replay_buffer_shards_colocated_with_driver": True,
        "rollout_fragment_length": sub_buffer_size
    }
    hyper_parameters["replay_buffer_config"] = replay_buffer_config
    hyper_parameters["train_batch_size"] = int(hyper_parameters["train_batch_size"] / sub_buffer_size)
    algorithm = ApexDDQNWithDPBER(config=hyper_parameters, env=settings.apex.env)

# Check path available
check_path(log_path)
log_path = path.join(log_path, run_name)
check_path(log_path)
check_path(checkpoint_path)
checkpoint_path = path.join(checkpoint_path, run_name)
check_path(checkpoint_path)

print(algorithm.config.to_dict()["replay_buffer_config"])

with open(os.path.join(checkpoint_path, "%s_config.pyl" % run_name), "wb") as f:
    _ = algorithm.config.to_dict()
    _.pop("multiagent")
    pickle.dump(_, f)

checkpoint_path = path.join(checkpoint_path, "results")
check_path(checkpoint_path)
mlflow.log_artifacts(checkpoint_path)

# Run algorithms
keys_to_extract = {"episode_reward_max", "episode_reward_min", "episode_reward_mean"}
for i in tqdm.tqdm(range(1, 10000)):
    try:
        result = algorithm.train()
        time_used = result["time_total_s"]
        # statistics
        sampler = result.get("sampler_results", None)
    except:
        continue
    try:
        if i >= 10 and i % settings.log.log == 0:
            learner_data = result["info"].copy()
            learner_data.pop("learner")
            logs_with_timeout(learner_data, step=result["episodes_total"])
            _save = {key: sampler[key] for key in keys_to_extract if key in sampler}
            logs_with_timeout(_save, step=result["episodes_total"])
        if i % settings.log.log == 0:
            algorithm.save_checkpoint(checkpoint_path)
    except FunctionTimedOut:
        tqdm.tqdm.write("logging failed")
    except MlflowException:
        tqdm.tqdm.write("logging failed")
    with open(path.join(log_path, str(i) + ".json"), "w") as f:
        result["config"] = None
        json.dump(convert_np_arrays(result), f)
    if time_used >= settings.log.max_time:
        break

mlflow.log_artifacts(log_path)
mlflow.log_artifacts(checkpoint_path)
