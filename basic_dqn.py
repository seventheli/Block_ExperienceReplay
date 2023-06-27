import os
import gym
import tqdm
import json
import mlflow
import pickle
import zipfile
import argparse
import datetime
from os import path
from dynaconf import Dynaconf
from ray.rllib.algorithms.dqn import DQN
from algorithms_with_statistics.basic_dqn import DQNWithLogging
from replay_buffer.ber import BlockReplayBuffer
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from utils import init_ray, check_path, logs_with_timeout, convert_np_arrays
from mlflow.exceptions import MlflowException
from func_timeout import FunctionTimedOut

checkpoint_path = "./checkpoint/"
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

# Set MLflow
mlflow.set_tracking_uri(settings.mlflow.url)
mlflow.set_experiment(experiment_name=settings.mlflow.experiment)
mlflow_client = mlflow.tracking.MlflowClient()

# Set hyper parameters
hyper_parameters = settings.dqn.hyper_parameters.to_dict()
if sub_buffer_size == 0:
    # Set run object
    run_name = "DQN_ER_" + datetime.datetime.now().strftime("%Y%m%d")
    mlflow_run = mlflow.start_run(run_name=run_name,
                                  tags={"mlflow.user": settings.mlflow.user})
    # Log parameters
    mlflow.log_params(hyper_parameters["replay_buffer_config"])
    mlflow.log_params({key: hyper_parameters[key] for key in hyper_parameters.keys() if key not in ["replay_buffer_config"]})
else:
    # Set run object
    run_name = "DQN_BER_" + datetime.datetime.now().strftime("%Y%m%d")
    mlflow_run = mlflow.start_run(run_name=run_name,
                                  tags={"mlflow.user": settings.mlflow.user})
    # Log parameters
    env_example = wrap_deepmind(gym.make(settings.dqn.env))
    mlflow.log_params({
        **settings.dqn.hyper_parameters.replay_buffer_config.to_dict(),
        "type": "BlockReplayBuffer",
        "sub_buffer_size": sub_buffer_size,
    })
    mlflow.log_params({key: hyper_parameters[key] for key in hyper_parameters.keys() if key not in ["replay_buffer_config"]})
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

# Check path available
log_path = path.join(settings.log.save_file, settings.dqn.env)
check_path(log_path)
log_path = path.join(log_path, run_name)
check_path(log_path)
checkpoint_path = path.join(settings.log.save_checkout, settings.dqn.env)
check_path(checkpoint_path)
checkpoint_path = path.join(checkpoint_path, run_name)
check_path(checkpoint_path)

with open(os.path.join(checkpoint_path, "%s config.pyl" % run_name), "wb") as f:
    _ = algorithm.config.to_dict()
    _.pop("multiagent")
    pickle.dump(_, f)
mlflow.log_artifacts(checkpoint_path)

# Run algorithms
keys_to_extract = {"episode_reward_max", "episode_reward_min", "episode_reward_mean"}
for i in tqdm.tqdm(range(1, 10000)):
    result = algorithm.train()
    time_used = result["time_total_s"]
    # statistics
    evaluation = result.get("evaluation", None)
    sampler = result.get("sampler_results", None)
    try:
        if evaluation is not None:
            _save = {"eval_" + key: evaluation[key] for key in keys_to_extract if key in evaluation}
            logs_with_timeout(_save, step=result["episodes_total"])
        if i >= 10 or i % settings.log.log == 0:
            learner_data = result["info"].copy()
            if learner_data["learner"].get("time_usage", None) is not None:
                logs_with_timeout(learner_data["learner"].get("time_usage"), step=result["episodes_total"])
            learner_data.pop("learner")
            logs_with_timeout(learner_data, step=result["episodes_total"])
            _save = {key: sampler[key] for key in keys_to_extract if key in sampler}
            logs_with_timeout(_save, step=result["episodes_total"])
        if i % (settings.log.log * 100) == 0:
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
with zipfile.ZipFile(os.path.join(log_path, '%s_log.zip' % run_name), 'w') as f:
    for file in os.listdir(log_path):
        f.write(os.path.join(log_path, file))
mlflow.log_artifacts(log_path)
