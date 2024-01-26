import os
import ray
import argparse
import json
import pickle
import tqdm
from os import path
from utils import check_path, convert_np_arrays
from dynaconf import Dynaconf
from ray.rllib.models import ModelCatalog
from model import CNN
from ray.tune.registry import register_env
from ray.tune.logger import JsonLogger
from algorithms.ddqn_pber import DDQNWithMPBER
from replay_buffer.mpber import MultiAgentPrioritizedBlockReplayBuffer

from utils import minigrid_env_creator as env_creator

ray.init(
    num_cpus=6, num_gpus=1,
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
env_name = parser.parse_args().env_path
run_name = str(parser.parse_args().run_name)
log_path = parser.parse_args().log_path
checkpoint_path = parser.parse_args().checkpoint_path
run_name = env_name + " pber " + run_name

# Check path available
check_path(log_path)
log_path = str(path.join(log_path, run_name))
check_path(log_path)
check_path(checkpoint_path)
checkpoint_path = path.join(checkpoint_path, run_name)
check_path(checkpoint_path)

setting = parser.parse_args().setting_path
setting = Dynaconf(envvar_prefix="DYNACONF", settings_files=setting)

hyper_parameters = setting.hyper_parameters.to_dict()
hyper_parameters["logger_config"] = {"type": JsonLogger, "logdir": checkpoint_path}
hyper_parameters["env_config"] = {
    "id": env_name,
    "size": 12,
    "roads": (1, 4),
    "max_steps": 300,
    "battery": 100,
    "img_size": 80,
    "tile_size": 8,
    "num_stack": 1,
    "render_mode": "rgb_array",
    "agent_pov": False
}

# Build env
env_example = env_creator(hyper_parameters["env_config"])
obs, _ = env_example.reset()
step = env_example.step(1)
print(env_example.action_space, env_example.observation_space)

register_env(env_name, env_creator)

ModelCatalog.register_custom_model("CNN", CNN)

# Set BER
sub_buffer_size = hyper_parameters["rollout_fragment_length"]
replay_buffer_config = {
    **setting.dqn.hyper_parameters.replay_buffer_config.to_dict(),
    "type": MultiAgentPrioritizedBlockReplayBuffer,
    "obs_space": env_example.observation_space,
    "action_space": env_example.action_space,
    "sub_buffer_size": sub_buffer_size,
    "worker_side_prioritization": False,
    "replay_sequence_length": 1,
    "rollout_fragment_length": hyper_parameters["rollout_fragment_length"]

}
hyper_parameters["replay_buffer_config"] = replay_buffer_config
hyper_parameters["train_batch_size"] = int(hyper_parameters["train_batch_size"] / sub_buffer_size)
trainer = DDQNWithMPBER(config=hyper_parameters, env=env_name)

checkpoint_path = str(checkpoint_path)
with open(os.path.join(checkpoint_path, "%s_config.pyl" % run_name), "wb") as f:
    pickle.dump(trainer.config.to_dict(), f)

checkpoint_path = str(path.join(checkpoint_path, "results"))
check_path(checkpoint_path)

# Run algorithms
for i in tqdm.tqdm(range(1, setting.log.max_run)):
    result = trainer.train()
    time_used = result["time_total_s"]
    if i % setting.log.log == 0:
        trainer.save_checkpoint(checkpoint_path)
    with open(path.join(log_path, str(i) + ".json"), "w") as f:
        result["config"] = None
        json.dump(convert_np_arrays(result), f)
    if time_used >= setting.log.max_time:
        break
