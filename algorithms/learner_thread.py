import time

from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.algorithms.dqn.learner_thread import LearnerThread as OriginalLearnerThread
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from itertools import chain
from utils import split_list_into_n_parts
import ray
import numpy as np
import zlib


def decompress_data(data_info):
    data, length, shape = data_info
    compressed_item = data[:length]
    decompressed_bytes = zlib.decompress(compressed_item)
    array = np.frombuffer(decompressed_bytes, dtype=np.uint8).reshape(shape)
    return array


@ray.remote(num_cpus=1, max_calls=200)
def decompress_data_loop(rows):
    _ = []
    for each in rows:
        _.append(decompress_data(each))
    return _


def parallel_decompress(obs, lengths_obs, shapes, new_obs, lengths_new_obs):
    data_info_list = list(zip(obs, lengths_obs, shapes)) + list(zip(new_obs, lengths_new_obs, shapes))
    data_info_list = split_list_into_n_parts(data_info_list, 4)
    result_ids = [decompress_data_loop.remote(batch) for batch in data_info_list]
    results = ray.get(result_ids)
    results = list(chain(*results))
    midpoint = len(results) // 2
    decompressed_obs, decompressed_new_obs = results[:midpoint], results[midpoint:]
    return decompressed_obs, decompressed_new_obs


def update_ma_batch(ma_batch):
    _ = time.time()
    obs = ma_batch["default_policy"]["obs"].reshape(len(ma_batch["default_policy"]["shape"]), -1)
    new_obs = ma_batch["default_policy"]["new_obs"].reshape(len(ma_batch["default_policy"]["shape"]), -1)
    decompressed_obs, decompressed_new_obs = parallel_decompress(
        obs, ma_batch["default_policy"]["length_obs"], ma_batch["default_policy"]["shape"],
        new_obs, ma_batch["default_policy"]["length_new_obs"]
    )
    obs = np.concatenate(decompressed_obs)
    new_obs = np.concatenate(decompressed_new_obs)
    data = SampleBatch(
        {
            "obs": obs,
            "new_obs": new_obs,
            "actions": ma_batch["default_policy"]["actions"],
            "rewards": ma_batch["default_policy"]["rewards"],
            "terminateds": ma_batch["default_policy"]["terminateds"],
            "truncateds": ma_batch["default_policy"]["truncateds"],
            "weights": ma_batch["default_policy"]["weights"],
            "batch_indexes": ma_batch["default_policy"]["batch_indexes"],
        }
    )
    updated_mab = MultiAgentBatch({"default_policy": data},
                                  env_steps=ma_batch["default_policy"].count)
    return updated_mab


class LearnerThread(OriginalLearnerThread):
    def __init__(self, local_worker):
        super().__init__(local_worker)

    def step(self):
        with self.overall_timer:
            with self.queue_timer:
                replay_actor, ma_batch = self.inqueue.get()
            if ma_batch is not None:
                prio_dict = {}
                ma_batch = update_ma_batch(ma_batch)
                # ma_batch = ray.get(ma_batch)
                with self.grad_timer:
                    # Use LearnerInfoBuilder as a unified way to build the
                    # final results dict from `learn_on_loaded_batch` call(s).
                    # This makes sure results dicts always have the same
                    # structure no matter the setup (multi-GPU, multi-agent,
                    # minibatch SGD, tf vs torch).
                    learner_info_builder = LearnerInfoBuilder(num_devices=1)
                    multi_agent_results = self.local_worker.learn_on_batch(ma_batch)
                    self.policy_ids_updated.extend(list(multi_agent_results.keys()))
                    for pid, results in multi_agent_results.items():
                        learner_info_builder.add_learn_on_batch_results(results, pid)
                        td_error = results["td_error"]
                        # Switch off auto-conversion from numpy to torch/tf
                        # tensors for the indices. This may lead to errors
                        # when sent to the buffer for processing
                        # (may get manipulated if they are part of a tensor).
                        ma_batch.policy_batches[pid].set_get_interceptor(None)
                        prio_dict[pid] = (
                            ma_batch.policy_batches[pid].get("batch_indexes"),
                            td_error,
                        )
                    self.learner_info = learner_info_builder.finalize()
                    self.grad_timer.push_units_processed(ma_batch.count)
                # Put tuple: replay_actor, prio-dict, env-steps, and agent-steps into
                # the queue.
                self.outqueue.put(
                    (replay_actor, prio_dict, ma_batch.count, ma_batch.agent_steps())
                )
                self.learner_queue_size.push(self.inqueue.qsize())
                self.overall_timer.push_units_processed(
                    ma_batch and ma_batch.count or 0
                )
                del ma_batch
