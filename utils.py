import os
import sys
import gc
import ray
import yaml
import numpy as np
from gym import spaces
from typing import Dict, Tuple, Union
from ray.tune.logger import UnifiedLogger

def custom_logger_creator(config):
    # `config` is a LoggerContext object.
    loggers = [UnifiedLogger(config, logdir=config["checkpoints"])]
    return loggers


def get_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    size = 0

    while obj_q:
        size += sum(sys.getsizeof(i) for i in obj_q)
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return size


def init_ray(ray_setting=None):
    if ray_setting is not None:
        with open(ray_setting, 'r') as file:
            settings = yaml.safe_load(file)
        ray.init(**settings)
    else:
        ray.init()


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def convert_np_arrays(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_np_arrays(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_arrays(item) for item in obj]
    else:
        return obj


# This function is copied from:
# https://github.com/DLR-RM/stable-baselines3/
def get_obs_shape(
        observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return 1,
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return int(len(observation_space.nvec)),
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return int(observation_space.n),
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}


# This function is copied from:
# https://github.com/DLR-RM/stable-baselines3/
def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
