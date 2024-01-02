import os
import ray
import json
import torch
import pickle
import tqdm
import argparse
import gymnasium as gym
from os import path
from model import CustomCNN
from dynaconf import Dynaconf
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.tune.logger import JsonLogger
from algorithms.apex_ddqn import ApexDDQNWithDPBER
from minigrid.wrappers import RGBImgPartialObsWrapper
from replay_buffer.mpber import MultiAgentPrioritizedBlockReplayBuffer
from minigrid.wrappers import ImgObsWrapper
from utils import check_path, convert_np_arrays

ray.init(
    num_cpus=16, num_gpus=1,
    include_dashboard=False,
    _system_config={"maximum_gcs_destroyed_actor_cached_count": 200},
)

parser = argparse.ArgumentParser()
parser.add_argument("-R", "--run_name", dest="run_name", type=int)
parser.add_argument("-S", "--setting", dest="setting_path", type=str)
parser.add_argument("-L", "--log_path", dest="log_path", type=str)
parser.add_argument("-C", "--checkpoint_path", dest="checkpoint_path", type=str)
parser.add_argument("-E", "--env", dest="env_path", type=str)

# Config path
env = "MiniGrid-" + parser.parse_args().env_path
run_name = parser.parse_args().run_name
log_path = parser.parse_args().log_path
checkpoint_path = parser.parse_args().checkpoint_path
setting = parser.parse_args().setting_path
setting = Dynaconf(envvar_prefix="DYNACONF", settings_files=setting)

hyper_parameters = setting.hyper_parameters.to_dict()
hyper_parameters["logger_config"] = {"type": JsonLogger, "logdir": checkpoint_path}
hyper_parameters["env_config"] = {
    "id": env

}
print("log path: %s, check_path: %s" % (log_path, checkpoint_path))


# Build env
def env_creator(env_config):
    _env = gym.make(env_config["id"], render_mode="rgb_array")
    _env = RGBImgPartialObsWrapper(_env, tile_size=env_config["tile_size"])
    return ImgObsWrapper(_env)


env = env_creator(hyper_parameters["env_config"])
obs, _ = env.reset()
step = env.step(1)
print(env.action_space, env.observation_space)

register_env("example", env_creator)

ModelCatalog.register_custom_model("CustomCNN", CustomCNN)

hyper_parameters["model"] = {
    "custom_model": "CustomCNN",
    "no_final_linear": True,
    "fcnet_hiddens": hyper_parameters["hiddens"],
    "custom_model_config": {},
}

# Set BER
sub_buffer_size = hyper_parameters["rollout_fragment_length"]
replay_buffer_config = {
    **hyper_parameters["replay_buffer_config"],
    "type": MultiAgentPrioritizedBlockReplayBuffer,
    "capacity": int(hyper_parameters["replay_buffer_config"]["capacity"]),
    "obs_space": env.observation_space,
    "action_space": env.action_space,
    "sub_buffer_size": sub_buffer_size,
    "worker_side_prioritization": False,
    "replay_buffer_shards_colocated_with_driver": True,
    "rollout_fragment_length": hyper_parameters["rollout_fragment_length"]
}
hyper_parameters["replay_buffer_config"] = replay_buffer_config
hyper_parameters["train_batch_size"] = int(hyper_parameters["train_batch_size"] / sub_buffer_size)

# Set trainer
trainer = ApexDDQNWithDPBER(config=hyper_parameters, env="example")

run_name = hyper_parameters["env_config"]["id"] + " dpber " + run_name

# Check path available
check_path(log_path)
log_path = path.join(log_path, run_name)
check_path(log_path)
check_path(checkpoint_path)
checkpoint_path = path.join(checkpoint_path, run_name)
check_path(checkpoint_path)

print(trainer.config.to_dict()["replay_buffer_config"])

with open(os.path.join(checkpoint_path, "%s_config.pyl" % run_name), "wb") as f:
    pickle.dump(trainer.config.to_dict(), f)

checkpoint_path = path.join(checkpoint_path, "results")
check_path(checkpoint_path)

# Run algorithms
for i in tqdm.tqdm(range(1, setting.max_run)):
    try:
        result = trainer.train()
        time_used = result["time_total_s"]
        if i % setting.log.log == 0:
            trainer.save_checkpoint(checkpoint_path)
        with open(path.join(log_path, str(i) + ".json"), "w") as f:
            result["config"] = None
            json.dump(convert_np_arrays(result), f)
        if time_used >= setting.log.max_time:
            break
    except:
        pass
