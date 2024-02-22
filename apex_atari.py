import os
import ray
import argparse
import json
import pickle
import tqdm
from os import path
from utils import check_path, convert_np_arrays, env_creator
from dynaconf import Dynaconf
from ray.tune.registry import register_env
from ray.tune.logger import JsonLogger
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig
from algorithms.apex_ddqn import ApexDDQNWithDPBER
from replay_buffer.mpber import MultiAgentPrioritizedBlockReplayBuffer

# Init Ray
ray.init(
    num_cpus=20, num_gpus=1,
    include_dashboard=False,
    _system_config={"maximum_gcs_destroyed_actor_cached_count": 200},
)

parser = argparse.ArgumentParser()
parser.add_argument("-R", "--run_name", dest="run_name", type=int)
parser.add_argument("-S", "--setting", dest="setting_path", type=str)
parser.add_argument("-L", "--log_path", dest="log_path", type=str)
parser.add_argument("-C", "--checkpoint_path", dest="checkpoint_path", type=str)
parser.add_argument("-E", "--env", dest="env_path", type=str)
parser.add_argument("-SBZ", "--sbz", dest="sub_buffer_size", type=int)

# Config path
env_name = parser.parse_args().env_path
sub_buffer_size = int(parser.parse_args().sub_buffer_size)
run_name = str(parser.parse_args().run_name)
log_path = parser.parse_args().log_path
checkpoint_path = parser.parse_args().checkpoint_path

# Check path available
check_path(log_path)
log_path = str(path.join(log_path, run_name))
check_path(log_path)
check_path(checkpoint_path)
checkpoint_path = path.join(checkpoint_path, run_name)
check_path(checkpoint_path)

# Set hyper parameters
setting = parser.parse_args().setting_path
setting = Dynaconf(envvar_prefix="DYNACONF", settings_files=setting)

hyper_parameters = setting.hyper_parameters.to_dict()
hyper_parameters["logger_config"] = {"type": JsonLogger, "logdir": checkpoint_path}
hyper_parameters["env_config"] = {
    "id": env_name,
}

env_example = env_creator(hyper_parameters["env_config"])
obs, _ = env_example.reset()
step = env_example.step(1)
print(env_example.action_space, env_example.observation_space)
register_env("example", env_creator)

print("log path: %s; check_path: %s" % (log_path, checkpoint_path))
if hyper_parameters["double_q"]:
    double_q = "DDQN"
else:
    double_q = "DQN"

if sub_buffer_size == 0:
    # Set run object
    run_name = "APEX_%s_%s" % (double_q, env_name) + "_DPER_%d" % run_name
    config = ApexDQNConfig().environment("example")
    config.update_from_dict(hyper_parameters)
    trainer = config.build()
else:
    # Set run object
    run_name = "APEX_DDQN_" + env_name + "_DPBER_%d" % run_name
    replay_buffer_config = {
        **hyper_parameters["replay_buffer_config"],
        "type": MultiAgentPrioritizedBlockReplayBuffer,
        "capacity": int(hyper_parameters["replay_buffer_config"]["capacity"]),
        "obs_space": env_example.observation_space,
        "action_space": env_example.action_space,
        "sub_buffer_size": sub_buffer_size,
        "worker_side_prioritization": False,
        "replay_buffer_shards_colocated_with_driver": True,
        "rollout_fragment_length": hyper_parameters["rollout_fragment_length"]
    }
    hyper_parameters["replay_buffer_config"] = replay_buffer_config
    hyper_parameters["train_batch_size"] = int(hyper_parameters["train_batch_size"] / sub_buffer_size)
    trainer = ApexDDQNWithDPBER(config=hyper_parameters, env="example")

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
