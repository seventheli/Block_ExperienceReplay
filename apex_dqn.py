import os
import gym
import tqdm
import json
import pickle
import argparse
import datetime
from os import path
from dynaconf import Dynaconf
from algorithms.apex_ddqn_pber import ApexDDQNWithDPBER
from replay_buffer.mpber import MultiAgentPrioritizedBlockReplayBuffer
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from utils import init_ray, check_path, convert_np_arrays
from ray.rllib.algorithms.apex_dqn import ApexDQN

init_ray("./ray_config.yml")

parser = argparse.ArgumentParser()
parser.add_argument("-S", "--setting", dest="setting_path", type=str)
parser.add_argument("-SBZ", "--sub_buffer_size", dest="sub_buffer_size", type=int, default=0)

sub_buffer_size = parser.parse_args().sub_buffer_size

# Config path
settings = parser.parse_args().setting_path
settings = Dynaconf(envvar_prefix="DYNACONF", settings_files=settings)

# Set hyper parameters
hyper_parameters = settings.apex.hyper_parameters.to_dict()
if hyper_parameters["double_q"]:
    run_name = "APEX_DDQN"
else:
    run_name = "APEX_DQN"

if sub_buffer_size == 0:
    # Set run object
    run_name = run_name + "_" + settings.dqn.env + "_DPER_" + datetime.datetime.now().strftime("%Y%m%d")
    algorithm = ApexDQN(config=hyper_parameters, env=settings.apex.env)
else:
    # Set run object
    run_name = run_name + "_" + settings.dqn.env + "_DPBER_" + datetime.datetime.now().strftime("%Y%m%d")
    env_example = wrap_deepmind(gym.make(settings.apex.env))
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
