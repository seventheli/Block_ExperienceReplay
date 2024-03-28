import os
import json
import tqdm
import pickle
from utils import check_path, convert_np_arrays, flatten_dict


def run_loop(trainer, max_run, log, log_path, max_time, checkpoint_path, run_name, use_normal=False):
    # Common setup
    checkpoint_path = str(checkpoint_path)
    # Save initial configuration
    with open(os.path.join(checkpoint_path, "%s_config.pyl" % run_name), "wb") as f:
        _ = trainer.config.to_dict()
        _.pop("multiagent")
        pickle.dump(_, f)

    # Prepare checkpoint path for results
    results_path = os.path.join(checkpoint_path, "results")
    check_path(results_path)

    # Run training loop
    for i in tqdm.tqdm(range(1, max_run + 1)):
        result = trainer.train()
        time_used = result["time_total_s"]

        # Save checkpoint conditionally
        if i % log == 0:
            trainer.save_checkpoint(results_path)

        # Process result for saving
        if use_normal:
            buf = flatten_dict(trainer.local_replay_buffer.stats())
            buf["est_size_gb"] = buf["est_size_bytes"] / 1e9
            result['buffer'] = buf

        result["config"] = None  # Clear config from result to avoid redundancy
        # Save training result
        with open(os.path.join(log_path, f"{i}.json"), "w") as f:
            json.dump(convert_np_arrays(result), f)

        # Break if exceeded maximum time allowed
        if time_used >= max_time:
            break
