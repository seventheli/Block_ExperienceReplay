# import time
#
# from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
# from ray.rllib.algorithms.dqn.learner_thread import LearnerThread as OriginalLearnerThread
# from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
# from itertools import chain
# from utils import split_list_into_n_parts
# import ray
# import numpy as np
# import zlib
#
#
# def decompress_data(data_info):
#     data, length, shape = data_info
#     compressed_item = data[:length]
#     decompressed_bytes = zlib.decompress(compressed_item)
#     array = np.frombuffer(decompressed_bytes, dtype=np.uint8).reshape(shape)
#     return array
#
#
# def parallel_decompress(obs, lengths_obs, shapes, new_obs, lengths_new_obs):
#     data_info_list = list(zip(obs, lengths_obs, shapes)) + list(zip(new_obs, lengths_new_obs, shapes))
#     results = []
#     for each in data_info_list:
#         results.append(decompress_data(each))
#     midpoint = len(results) // 2
#     decompressed_obs, decompressed_new_obs = results[:midpoint], results[midpoint:]
#     return decompressed_obs, decompressed_new_obs
#
#
# def decompress_sample_batch(replay_actor, ma_batch):
#     obs = ma_batch["default_policy"]["obs"].reshape(len(ma_batch["default_policy"]["shape"]), -1)
#     new_obs = ma_batch["default_policy"]["new_obs"].reshape(len(ma_batch["default_policy"]["shape"]), -1)
#     decompressed_obs, decompressed_new_obs = parallel_decompress(
#         obs, ma_batch["default_policy"]["length_obs"], ma_batch["default_policy"]["shape"],
#         new_obs, ma_batch["default_policy"]["length_new_obs"]
#     )
#     obs = np.concatenate(decompressed_obs)
#     new_obs = np.concatenate(decompressed_new_obs)
#     data = SampleBatch(
#         {
#             "obs": obs,
#             "new_obs": new_obs,
#             "actions": ma_batch["default_policy"]["actions"],
#             "rewards": ma_batch["default_policy"]["rewards"],
#             "terminateds": ma_batch["default_policy"]["terminateds"],
#             "truncateds": ma_batch["default_policy"]["truncateds"],
#             "weights": ma_batch["default_policy"]["weights"],
#             "batch_indexes": ma_batch["default_policy"]["batch_indexes"],
#         }
#     )
#     updated_mab = MultiAgentBatch({"default_policy": data},
#                                   env_steps=ma_batch["default_policy"].count)
#     return replay_actor, updated_mab
#
#
# @ray.remote(num_cpus=1, max_calls=50, num_returns=1)
# def update_ma_batch_loop(samples):
#     _ = []
#     for each in samples:
#         _.append(decompress_sample_batch(*each))
#     return _
#
#
# class LearnerThread(OriginalLearnerThread):
#     def __init__(self, local_worker, sub_size=100, split_size=5):
#         super().__init__(local_worker)
#         self.local_store = []
#         self.limit = sub_size
#         self.split_size = split_size
#
#     # def step(self):
#     #     with self.overall_timer:
#     #         # Read
#     #         with self.queue_timer:
#     #             replay_actor, ma_batch = self.inqueue.get()
#     #         if ma_batch is not None:
#     #             self.local_store.append([replay_actor, ma_batch])
#     #         if len(self.local_store) == self.limit:
#     #             ma_batches = split_list_into_n_parts(self.local_store, self.split_size)
#     #             ma_batches = [update_ma_batch_loop.remote(i) for i in ma_batches]
#     #             ma_batches = ray.get(ma_batches)
#     #             ma_batches = list(chain(*ma_batches))
#     #             for (replay_actor, ma_batch) in ma_batches:
#     #                 prio_dict = {}
#     #                 with self.grad_timer:
#     #                     # Use LearnerInfoBuilder as a unified way to build the
#     #                     # final results dict from `learn_on_loaded_batch` call(s).
#     #                     # This makes sure results dicts always have the same
#     #                     # structure no matter the setup (multi-GPU, multi-agent,
#     #                     # minibatch SGD, tf vs torch).
#     #                     learner_info_builder = LearnerInfoBuilder(num_devices=1)
#     #                     multi_agent_results = self.local_worker.learn_on_batch(ma_batch)
#     #                     self.policy_ids_updated.extend(list(multi_agent_results.keys()))
#     #                     for pid, results in multi_agent_results.items():
#     #                         learner_info_builder.add_learn_on_batch_results(results, pid)
#     #                         td_error = results["td_error"]
#     #                         # Switch off auto-conversion from numpy to torch/tf
#     #                         # tensors for the indices. This may lead to errors
#     #                         # when sent to the buffer for processing
#     #                         # (may get manipulated if they are part of a tensor).
#     #                         ma_batch.policy_batches[pid].set_get_interceptor(None)
#     #                         prio_dict[pid] = (
#     #                             ma_batch.policy_batches[pid].get("batch_indexes"),
#     #                             td_error,
#     #                         )
#     #                     self.learner_info = learner_info_builder.finalize()
#     #                     self.grad_timer.push_units_processed(ma_batch.count)
#     #                 # Put tuple: replay_actor, prio-dict, env-steps, and agent-steps into
#     #                 # the queue.
#     #                 self.outqueue.put(
#     #                     (replay_actor, prio_dict, ma_batch.count, ma_batch.agent_steps())
#     #                 )
#     #                 self.learner_queue_size.push(self.inqueue.qsize())
#     #                 self.overall_timer.push_units_processed(
#     #                     ma_batch and ma_batch.count or 0
#     #                 )
#     #                 del ma_batch
#     #             self.local_store = []
