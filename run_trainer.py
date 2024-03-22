import os
import json
import tqdm
import pickle
from utils import check_path, convert_np_arrays, flatten_dict


def run_loop(trainer, max_run, log, log_path, max_time, checkpoint_path, run_name):
    checkpoint_path = str(checkpoint_path)
    with open(os.path.join(checkpoint_path, "%s_config.pyl" % run_name), "wb") as f:
        pickle.dump(trainer.config.to_dict(), f)

    checkpoint_path = str(os.path.join(checkpoint_path, "results"))
    check_path(checkpoint_path)

    # Run algorithms
    for i in tqdm.tqdm(range(1, max_run)):
        result = trainer.train()
        time_used = result["time_total_s"]
        if i % log == 0:
            trainer.save_checkpoint(checkpoint_path)
        with open(os.path.join(log_path, str(i) + ".json"), "w") as f:
            result["config"] = None
            json.dump(convert_np_arrays(result), f)
        if time_used >= max_time:
            break


def run_loop_single(trainer, max_run, log, log_path, max_time, checkpoint_path, run_name):
    checkpoint_path = str(checkpoint_path)
    with open(os.path.join(checkpoint_path, "%s_config.pyl" % run_name), "wb") as f:
        pickle.dump(trainer.config.to_dict(), f)

    checkpoint_path = str(os.path.join(checkpoint_path, "results"))
    check_path(checkpoint_path)

    # Run algorithms
    for i in tqdm.tqdm(range(1, max_run)):
        result = trainer.train()
        time_used = result["time_total_s"]
        if i % log == 0:
            trainer.save_checkpoint(checkpoint_path)
        with open(os.path.join(log_path, str(i) + ".json"), "w") as f:
            buf = flatten_dict(trainer.local_replay_buffer.stats())
            buf["est_size_gb"] = buf["est_size_bytes"] / 1e9
            result["config"] = None
            result['buffer'] = buf
            json.dump(convert_np_arrays(result), f)
        if time_used >= max_time:
            break
