from ray.rllib.utils.typing import EnvCreator
from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.utils.typing import AlgorithmConfigDict
import copy
import platform
from collections import defaultdict

import ray
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from algorithms.learner_thread import LearnerThread
from ray.rllib.algorithms.dqn.learner_thread import LearnerThread as OriginalLearnerThread
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.actors import create_colocated_actors
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_TRAINED,
)
from ray.tune.trainable import Trainable


class ApexDDQNWithDPBER(ApexDQN):

    def _init(self, config: AlgorithmConfigDict, env_creator: EnvCreator) -> None:
        super(ApexDDQNWithDPBER, self)._init(config, env_creator)

    # @override(Trainable)
    # def setup(self, config: AlgorithmConfig):
    #     super().setup(config)
    #
    #     num_replay_buffer_shards = self.config.optimizer["num_replay_buffer_shards"]
    #
    #     # Create copy here so that we can modify without breaking other logic
    #     replay_actor_config = copy.deepcopy(self.config.replay_buffer_config)
    #
    #     replay_actor_config["capacity"] = (
    #             self.config.replay_buffer_config["capacity"] // num_replay_buffer_shards
    #     )
    #
    #     ReplayActor = ray.remote(num_cpus=0, max_restarts=-1)(
    #         replay_actor_config["type"]
    #     )
    #
    #     # Place all replay buffer shards on the same node as the learner
    #     # (driver process that runs this execution plan).
    #     if replay_actor_config["replay_buffer_shards_colocated_with_driver"]:
    #         _replay_actors = create_colocated_actors(
    #             actor_specs=[  # (class, args, kwargs={}, count)
    #                 (
    #                     ReplayActor,
    #                     None,
    #                     replay_actor_config,
    #                     num_replay_buffer_shards,
    #                 )
    #             ],
    #             node=platform.node(),  # localhost
    #         )[
    #             0
    #         ]  # [0]=only one item in `actor_specs`.
    #     # Place replay buffer shards on any node(s).
    #     else:
    #         _replay_actors = [
    #             ReplayActor.remote(*replay_actor_config)
    #             for _ in range(num_replay_buffer_shards)
    #         ]
    #     self._replay_actor_manager = FaultTolerantActorManager(
    #         _replay_actors,
    #         max_remote_requests_in_flight_per_actor=(
    #             self.config.max_requests_in_flight_per_replay_worker
    #         ),
    #     )
    #     self._replay_req_timeout_s = self.config.timeout_s_replay_manager
    #     self._sample_req_tiemeout_s = self.config.timeout_s_sampler_manager
    #     if self.config.get("ram_saver", False):
    #         self.learner_thread = LearnerThread(self.workers.local_worker())
    #     else:
    #         self.learner_thread = OriginalLearnerThread(self.workers.local_worker())
    #
    #     self.learner_thread.start()
    #     self.steps_since_update = defaultdict(int)
    #     weights = self.workers.local_worker().get_weights()
    #     self.curr_learner_weights = ray.put(weights)
    #     self.curr_num_samples_collected = 0
    #     self._num_ts_trained_since_last_target_update = 0

    def update_replay_sample_priority(self) -> None:
        """Update the priorities of the sample batches with new priorities that are
        computed by the learner thread.
        """
        sub_buffer_size = self.config["replay_buffer_config"]["sub_buffer_size"]
        num_samples_trained_this_itr = 0
        for _ in range(self.learner_thread.outqueue.qsize()):
            if self.learner_thread.is_alive():
                (
                    replay_actor_id,
                    priority_dict,
                    env_steps,
                    agent_steps,
                ) = self.learner_thread.outqueue.get(timeout=0.001)
                # The different with original ApexDQN object
                for i in priority_dict:
                    batch_indices = priority_dict[i][0].reshape(-1, sub_buffer_size)[:, 0]
                    td_error = priority_dict[i][1].reshape([-1, sub_buffer_size]).mean(axis=1)
                    priority_dict[i] = (batch_indices, td_error)
                if self.config["replay_buffer_config"].get("prioritized_replay_alpha") > 0:
                    self._replay_actor_manager.foreach_actor(
                        func=lambda actor: actor.update_priorities(priority_dict),
                        remote_actor_ids=[replay_actor_id],
                        timeout_seconds=0,  # Do not wait for results.
                    )
                num_samples_trained_this_itr += env_steps
                self.update_target_networks(env_steps)
                self._counters[NUM_ENV_STEPS_TRAINED] += env_steps
                self._counters[NUM_AGENT_STEPS_TRAINED] += agent_steps
                self.workers.local_worker().set_global_vars(
                    {"timestep": self._counters[NUM_ENV_STEPS_TRAINED]}
                )
            else:
                raise RuntimeError("The learner thread died while training")

        self._timers["learner_dequeue"] = self.learner_thread.queue_timer
        self._timers["learner_grad"] = self.learner_thread.grad_timer
        self._timers["learner_overall"] = self.learner_thread.overall_timer

    @staticmethod
    def execution_plan(workers, config, **kwargs):
        pass
