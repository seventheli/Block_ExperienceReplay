import os
import gym
import tqdm
import json
import mlflow
import pickle
import argparse
import datetime
from os import path
from dynaconf import Dynaconf
from ray.rllib.algorithms.dqn import DQN
from algorithms_with_statistics.ddqn_pber import DDQNWithMPBERAndLogging
from algorithms_with_statistics.ddqn_per import DDQNWithMPERAndLogging
from algorithms.ddqn_pber import DDQNWithMPBER
from replay_buffer.mpber import MultiAgentPrioritizedBlockReplayBuffer
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from utils import init_ray, check_path, logs_with_timeout, convert_np_arrays
from mlflow.exceptions import MlflowException
from func_timeout import FunctionTimedOut

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

# Set hyper parameters
hyper_parameters = settings.dqn.hyper_parameters.to_dict()
if sub_buffer_size == 0:
    # Set run object
    run_name = "DDQN_PER_%s_%s" % (settings.dqn.env, datetime.datetime.now().strftime("%Y%m%d"))
    if with_er_logging:
        algorithm = DDQNWithMPERAndLogging(config=hyper_parameters, env=settings.dqn.env)
    else:
        algorithm = DQN(config=hyper_parameters, env=settings.dqn.env)
else:
    # Set run object
    run_name = "DDQN_PBER_%s_%s" % (settings.dqn.env, datetime.datetime.now().strftime("%Y%m%d"))
    # Log parameters
    env_example = wrap_deepmind(gym.make(settings.dqn.env))
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
        algorithm = DDQNWithMPBERAndLogging(config=hyper_parameters, env=settings.dqn.env)
    else:
        algorithm = DDQNWithMPBER(config=hyper_parameters, env=settings.dqn.env)

# Check path available
check_path(settings.log.save_file)
log_path = path.join(settings.log.save_file, run_name)
check_path(log_path)
check_path(settings.log.save_checkout)
checkpoint_path = path.join(settings.log.save_checkout, run_name)
check_path(checkpoint_path)

with open(os.path.join(checkpoint_path, "%s_config.pyl" % run_name), "wb") as f:
    _ = algorithm.config.to_dict()
    _.pop("multiagent")
    pickle.dump(_, f)

# Run algorithms
keys_to_extract = {"episode_reward_max", "episode_reward_min", "episode_reward_mean"}
for i in tqdm.tqdm(range(1, 10000)):
    try:
        result = algorithm.train()
        time_used = result["time_total_s"]
        if i % settings.log.log == 0:
            algorithm.save_checkpoint(checkpoint_path)
        with open(path.join(log_path, str(i) + ".json"), "w") as f:
            result["config"] = None
            json.dump(convert_np_arrays(result), f)
        if time_used >= settings.log.max_time:
            break
    except:
        pass