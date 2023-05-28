import tqdm
import json
import mlflow
import argparse
import datetime
from os import path
from dynaconf import Dynaconf
from utils import init_ray, check_path, logs_with_timeout, convert_np_arrays
from ray.rllib.algorithms.dqn import DQN

checkpoint_path = "./checkpoint/"
init_ray("./ray_config.yml")

parser = argparse.ArgumentParser()
parser.add_argument("-S", "--setting", dest="setting_path", type=str)
parser.add_argument("-BZ", "--batch_size", dest="batch_size", type=int, default=0)

# Config path
setting_path = parser.parse_args().setting_path
settings = Dynaconf(envvar_prefix="DYNACONF", settings_files=setting_path)

# Check path available
log_path = path.join(settings.log.save_file, settings.dqn.env)
check_path(log_path)
log_path = path.join(log_path, "DQN")
check_path(log_path)

# Attributes
batch_size = parser.parse_args().batch_size
if batch_size != 0:
    settings["dqn"]["hyper_parameters"]["train_batch_size"] = batch_size

# Set mlflow
mlflow.set_tracking_uri(settings.mlflow.url)
mlflow.set_experiment(experiment_name=settings.mlflow.experiment)
mlflow_client = mlflow.tracking.MlflowClient()
mlflow_run = mlflow.start_run(run_name="DQN_ER_" + datetime.datetime.now().strftime("%Y%m%d"),
                              tags={"mlflow.user": settings.mlflow.user})

# Set algorithm
hyper_parameters = settings.dqn.hyper_parameters.to_dict()
mlflow.log_params({key: hyper_parameters[key] for key in hyper_parameters.keys() if key not in ["replay_buffer_config"]})
mlflow.log_params(hyper_parameters["replay_buffer_config"])
algorithm = DQN(config=hyper_parameters, env=settings.dqn.env)

keys_to_extract = {"episode_reward_max", "episode_reward_min", "episode_reward_mean"}
for i in tqdm.tqdm(range(1, 10000)):
    result = algorithm.train()
    # statistics
    evaluation = result.get("evaluation", None)
    sampler = result.get("sampler_results", None)
    if i == 10 or i % settings.log.log == 0:
        learner_data = result["info"].copy()
        learner_data.pop("learner")
        if learner_data is not None:
            logs_with_timeout(learner_data)
            _save = {key: sampler[key] for key in keys_to_extract if key in sampler}
            logs_with_timeout(_save)
    if evaluation is not None:
        _save = {"eval_" + key: evaluation[key] for key in keys_to_extract if key in evaluation}
        logs_with_timeout(_save)
    with open(path.join(log_path, str(i) + ".json"), "w") as f:
        result["config"] = None
        json.dump(convert_np_arrays(result), f)
