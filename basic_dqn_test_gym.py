import os
import gym
import tqdm
import json
import pickle
import argparse
import datetime
from os import path
from dynaconf import Dynaconf
from ray.rllib.algorithms.dqn import DQN
from algorithms_with_statistics.basic_dqn import DQNWithLogging
from replay_buffer.ber import BlockReplayBuffer
from utils import init_ray, convert_np_arrays, check_path

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
    run_name = "DQN_ER_%s_%s" % (settings.dqn.env, datetime.datetime.now().strftime("%Y%m%d"))
else:
    # Set run object
    run_name = "DQN_BER_%s_%s" % (settings.dqn.env, datetime.datetime.now().strftime("%Y%m%d"))
    # Log parameters
    env_example = gym.make(settings.dqn.env)
    # Set BER
    replay_buffer_config = {
        **settings.dqn.hyper_parameters.replay_buffer_config.to_dict(),
        "storage_unit": "fragments",
        "type": BlockReplayBuffer,
        "obs_space": env_example.observation_space,
        "action_space": env_example.action_space,
        "sub_buffer_size": sub_buffer_size,
    }
    hyper_parameters["replay_buffer_config"] = replay_buffer_config

if with_er_logging:
    algorithm = DQNWithLogging(config=hyper_parameters, env=settings.dqn.env)
else:
    algorithm = DQN(config=hyper_parameters, env=settings.dqn.env)

print(algorithm.config.to_dict()["replay_buffer_config"])

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
        import tree
        import sys
        import numpy as np
        x = tree.flatten(algorithm.local_replay_buffer._storage[10])
        [v.nbytes if isinstance(v, np.ndarray) else sys.getsizeof(v) for v in x]
        print(sum([v.nbytes if isinstance(v, np.ndarray) else sys.getsizeof(v) for v in x]))
        import pdb
        pdb.set_trace()
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
