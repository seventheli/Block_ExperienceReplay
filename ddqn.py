import argparse
import datetime
import gym
import json
import mlflow
import os
import pickle
import torch
import tqdm
from algorithms.ddqn_pber import DDQNWithMPBER
from algorithms_with_statistics.ddqn_pber import DDQNWithMPBERAndERLogging
from algorithms_with_statistics.ddqn_per import DDQNWithMPERAndLogging
from dynaconf import Dynaconf
from func_timeout import FunctionTimedOut
from mlflow.exceptions import MlflowException
from os import path
from ray.rllib.algorithms.dqn import DQN
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
parser.add_argument("-E", "--er_logging", dest="er_logging", type=int, default=0)

if parser.parse_args().single_ray == 1:
    init_ray()
else:
    init_ray("./ray_config.yml")

sub_buffer_size = parser.parse_args().sub_buffer_size

# Configs
log_path = parser.parse_args().log_path
checkpoint_path = parser.parse_args().checkpoint_path
with_er_logging = parser.parse_args().er_logging
settings = parser.parse_args().setting_path
settings = Dynaconf(envvar_prefix="DYNACONF", settings_files=settings)

# Set MLflow
mlflow.set_tracking_uri(settings.mlflow.url)
mlflow.set_experiment(experiment_name=settings.mlflow.experiment)
mlflow_client = mlflow.tracking.MlflowClient()

# Set hyper parameters

hyper_parameters = settings.dqn.hyper_parameters.to_dict()
hyper_parameters["logger_config"] = {"type": UnifiedLogger, "logdir": checkpoint_path}
print("log path: %s \n check_path: %s" % (log_path, checkpoint_path))

if sub_buffer_size == 0:
    # Set run object
    run_name = settings.dqn.env + "_PER_%d" % parser.parse_args().run_name
    mlflow_run = mlflow.start_run(run_name=run_name,
                                  tags={"mlflow.user": settings.mlflow.user})
    # Log parameters
    mlflow.log_params(hyper_parameters["replay_buffer_config"])
    mlflow.log_params(
        {key: hyper_parameters[key] for key in hyper_parameters.keys() if key not in ["replay_buffer_config"]})
    if with_er_logging:
        algorithm = DDQNWithMPERAndLogging(config=hyper_parameters, env=settings.dqn.env)
    else:
        algorithm = DQN(config=hyper_parameters, env=settings.dqn.env)
else:
    # Set run object
    run_name = settings.dqn.env + "_PBER_%d" % parser.parse_args().run_name
    mlflow_run = mlflow.start_run(run_name=run_name,
                                  tags={"mlflow.user": settings.mlflow.user})
    env_example = wrap_deepmind(gym.make(settings.dqn.env))
    # Log parameters
    mlflow.log_params({
        **settings.dqn.hyper_parameters.replay_buffer_config.to_dict(),
        "type": "MultiAgentPrioritizedBlockReplayBuffer",
        "sub_buffer_size": sub_buffer_size,
    })
    mlflow.log_params(
        {key: hyper_parameters[key] for key in hyper_parameters.keys() if key not in ["replay_buffer_config"]})
    # Set BER
    replay_buffer_config = {
        **settings.dqn.hyper_parameters.replay_buffer_config.to_dict(),
        "type": MultiAgentPrioritizedBlockReplayBuffer,
        "obs_space": env_example.observation_space,
        "action_space": env_example.action_space,
        "sub_buffer_size": sub_buffer_size,
        "worker_side_prioritization": False,
        "replay_sequence_length": 1,
    }
    hyper_parameters["replay_buffer_config"] = replay_buffer_config
    if with_er_logging:
        algorithm = DDQNWithMPBERAndERLogging(config=hyper_parameters, env=settings.dqn.env)
    else:
        algorithm = DDQNWithMPBER(config=hyper_parameters, env=settings.dqn.env)

print(algorithm.config.to_dict()["replay_buffer_config"])

# Check path available
check_path(log_path)
log_path = path.join(log_path, run_name)
check_path(log_path)
check_path(checkpoint_path)
checkpoint_path = path.join(checkpoint_path, run_name)
check_path(checkpoint_path)

with open(os.path.join(checkpoint_path, "%s_config.pyl" % run_name), "wb") as f:
    _ = algorithm.config.to_dict()
    _.pop("multiagent")
    pickle.dump(_, f)

checkpoint_path = path.join(checkpoint_path, "results")
check_path(checkpoint_path)
mlflow.log_artifacts(checkpoint_path)

# Run algorithms
keys_to_extract_sam = {"episode_reward_max", "episode_reward_min", "episode_reward_mean"}
keys_to_extract_sta = {"num_agent_steps_sampled", "num_agent_steps_trained", "episode_reward_mean"}
keys_to_extract_buf = {"add_batch_time_ms", "replay_time_ms", "update_priorities_time_ms"}
for i in tqdm.tqdm(range(1, 10000)):
    try:
        result = algorithm.train()
        time_used = result["time_total_s"]
    except:
        continue
    try:
        if i >= settings.log.log and i % settings.log.log == 0:
            sampler = result.get("sampler_results", {}).copy()
            info = result.get("info", {}).copy()
            sam = {key: sampler[key] for key in keys_to_extract_sam if key in sampler}
            sta = {key: info[key] for key in keys_to_extract_sta if key in info}
            buf = flatten_dict(info.get("replay_shard_0", {}))
            lea = info.get("learner", {}).get("time_usage", {})
            mlflow.log_metrics({**sam, **sta, **buf, **lea}, step=result["episodes_total"])
            algorithm.save_checkpoint(checkpoint_path)
        if i % (settings.log.log * 100) == 0:
            mlflow.log_artifacts(log_path)
            mlflow.log_artifacts(checkpoint_path)
    except FunctionTimedOut:
        tqdm.tqdm.write("logging failed")
    except MlflowException:
        tqdm.tqdm.write("logging failed")
    with open(path.join(log_path, str(i) + ".json"), "w") as f:
        result["config"] = None
        json.dump(convert_np_arrays(result), f)
    if time_used >= settings.log.max_time:
        break

