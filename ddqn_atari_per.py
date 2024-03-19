import os
import ray
import argparse
from dynaconf import Dynaconf
from run_trainer import run_loop
from ray.tune.logger import JsonLogger
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
from utils import check_path, env_creator

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

# Config path
env_name = parser.parse_args().env_path
run_name = str(parser.parse_args().run_name)
log_path = parser.parse_args().log_path
checkpoint_path = parser.parse_args().checkpoint_path
run_name = "DDQN_%s" % env_name + "_PER_%s" % run_name

# Check path available
check_path(log_path)
log_path = str(os.path.join(log_path, run_name))
check_path(log_path)
check_path(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, run_name)
check_path(checkpoint_path)

setting = parser.parse_args().setting_path
setting = Dynaconf(envvar_prefix="DYNACONF", settings_files=setting)

hyper_parameters = setting.hyper_parameters.to_dict()
hyper_parameters["logger_config"] = {"type": JsonLogger, "logdir": checkpoint_path}
hyper_parameters["optimizer"] = {"num_replay_buffer_shards": 10}

# Build env
hyper_parameters["env_config"] = {
    "id": env_name,
}

env_example = env_creator(hyper_parameters["env_config"])
obs, _ = env_example.reset()
step = env_example.step(1)
print(env_example.action_space, env_example.observation_space)
register_env("example", env_creator)
print("log path: %s; check_path: %s" % (log_path, checkpoint_path))

# Set trainer
config = DQNConfig().environment("example")
config.update_from_dict(hyper_parameters)
trainer = config.build()

run_loop(trainer=trainer,
         log=setting.log.log,
         max_run=setting.log.max_run,
         max_time=setting.log.max_time,
         checkpoint_path=checkpoint_path,
         log_path=log_path,
         run_name=run_name)
