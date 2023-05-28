import tqdm
import json
import mlflow
import argparse
import datetime
import collections
from os import path
from dynaconf import Dynaconf
from utils import init_ray, check_path, log_with_timeout
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
    settings["ddqn"]["hyper_parameters"]["train_batch_size"] = batch_size

# Set mlflow
mlflow.set_tracking_uri(settings.mlflow.url)
mlflow.set_experiment(experiment_name=settings.mlflow.experiment)
mlflow_client = mlflow.tracking.MlflowClient()
mlflow_run = mlflow.start_run(run_name="DQN" + datetime.datetime.now().strftime("%Y%m%d"),
                              tags={"mlflow.user": settings.mlflow.user})

# Set algorithm
hyper_parameters = settings.dqn.hyper_parameters.to_dict()
algorithm = DQN(config=hyper_parameters, env=settings.dqn.env)

for i in tqdm.tqdm(range(1, 10000)):
    result = algorithm.train()
    if i == 10 or i % settings.log.log == 0:
        learner_data = result['info']['learner']
        if learner_data.get("time_usage", None) is not None:
            log_with_timeout()
    with open(path.join(log_path, str(i) + ".json"), "w") as f:
        result["config"] = None
        json.dump(result, f)
