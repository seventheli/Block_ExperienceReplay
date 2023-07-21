import os
import gym
import tqdm
import json
import pickle
import datetime
from os import path
from dynaconf import Dynaconf
from algorithms_with_statistics.basic_dqn import DQNWithLogging
from replay_buffer.ber import BlockReplayBuffer
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from utils import init_ray, convert_np_arrays, check_path

check_path("/tmp/")
init_ray("./ray_config.yml")

settings = "./settings/Test_er.yml"
settings = Dynaconf(envvar_prefix="DYNACONF", settings_files=settings)

hyper_parameters = settings.dqn.hyper_parameters.to_dict()
run_name = "DQN_BER_%s_%s" % (settings.dqn.env, datetime.datetime.now().strftime("%Y%m%d"))
# Log parameters
env_example = wrap_deepmind(gym.make(settings.dqn.env))
# Set BER
replay_buffer_config = {
    **settings.dqn.hyper_parameters.replay_buffer_config.to_dict(),
    "storage_unit": "fragments",
    "type": BlockReplayBuffer,
    "obs_space": env_example.observation_space,
    "action_space": env_example.action_space,
    "sub_buffer_size": 4,
}
hyper_parameters["replay_buffer_config"] = replay_buffer_config
algorithm = DQNWithLogging(config=hyper_parameters, env=settings.dqn.env)

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
