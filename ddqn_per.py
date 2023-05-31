import tqdm
import json
import mlflow
import argparse
import datetime
from os import path
from dynaconf import Dynaconf
from ray.rllib.algorithms.dqn import DQN
from utils import init_ray, check_path, logs_with_timeout, convert_np_arrays

checkpoint_path = "./checkpoint/"
init_ray("./ray_config.yml")

parser = argparse.ArgumentParser()
parser.add_argument("-S", "--setting", dest="setting_path", type=str)

# Config path
setting_path = parser.parse_args().setting_path
settings = Dynaconf(envvar_prefix="DYNACONF", settings_files=setting_path)

# Set algorithm
hyper_parameters = settings.dqn.hyper_parameters.to_dict()
mlflow.log_params({key: hyper_parameters[key] for key in hyper_parameters.keys() if key not in ["replay_buffer_config"]})
mlflow.log_params(hyper_parameters["replay_buffer_config"])
algorithm = DQN(config=hyper_parameters, env=settings.dqn.env)

# Set mlflow
run_name = "DQN_" + datetime.datetime.now().strftime("%Y%m%d")
mlflow.set_tracking_uri(settings.mlflow.url)
mlflow.set_experiment(experiment_name=settings.mlflow.experiment)
mlflow_client = mlflow.tracking.MlflowClient()
mlflow_run = mlflow.start_run(run_name=run_name,
                              tags={"mlflow.user": settings.mlflow.user})

# Check path available
log_path = path.join(settings.log.save_file, settings.dqn.env)
check_path(log_path)
log_path = path.join(log_path, run_name)
check_path(log_path)

keys_to_extract = {"episode_reward_max", "episode_reward_min", "episode_reward_mean"}
for i in tqdm.tqdm(range(1, 10000)):
    result = algorithm.train()
    # statistics
    evaluation = result.get("evaluation", None)
    sampler = result.get("sampler_results", None)
    if i >= 10 or i % settings.log.log == 0:
        learner_data = result["info"].copy()
        if learner_data["learner"].get("time_usage", None) is not None:
            logs_with_timeout(learner_data["learner"].get("time_usage"), step=result["episodes_total"])
        learner_data.pop("learner")
        logs_with_timeout(learner_data, step=result["episodes_total"])
        _save = {key: sampler[key] for key in keys_to_extract if key in sampler}
        logs_with_timeout(_save, step=result["episodes_total"])
    if evaluation is not None:
        _save = {"eval_" + key: evaluation[key] for key in keys_to_extract if key in evaluation}
        logs_with_timeout(_save, step=result["episodes_total"])
    with open(path.join(log_path, str(i) + ".json"), "w") as f:
        result["config"] = None
        json.dump(convert_np_arrays(result), f)
