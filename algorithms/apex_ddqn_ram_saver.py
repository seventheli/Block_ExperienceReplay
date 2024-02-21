from algorithms.apex_ddqn import ApexDDQNWithDPBER
import copy
import platform
from collections import defaultdict

import ray
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from algorithms.learner_thread import LearnerThread
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.actors import create_colocated_actors
from ray.rllib.utils.annotations import override

from ray.tune.trainable import Trainable


class ApexDDQNWithDPBER(ApexDDQNWithDPBER):

    @override(Trainable)
    def setup(self, config: AlgorithmConfig):
        super().setup(config)

        num_replay_buffer_shards = self.config.optimizer["num_replay_buffer_shards"]

        # Create copy here so that we can modify without breaking other logic
        replay_actor_config = copy.deepcopy(self.config.replay_buffer_config)

        replay_actor_config["capacity"] = (
                self.config.replay_buffer_config["capacity"] // num_replay_buffer_shards
        )

        ReplayActor = ray.remote(num_cpus=0, max_restarts=-1)(
            replay_actor_config["type"]
        )

        # Place all replay buffer shards on the same node as the learner
        # (driver process that runs this execution plan).
        if replay_actor_config["replay_buffer_shards_colocated_with_driver"]:
            _replay_actors = create_colocated_actors(
                actor_specs=[  # (class, args, kwargs={}, count)
                    (
                        ReplayActor,
                        None,
                        replay_actor_config,
                        num_replay_buffer_shards,
                    )
                ],
                node=platform.node(),  # localhost
            )[
                0
            ]  # [0]=only one item in `actor_specs`.
        # Place replay buffer shards on any node(s).
        else:
            _replay_actors = [
                ReplayActor.remote(*replay_actor_config)
                for _ in range(num_replay_buffer_shards)
            ]
        self._replay_actor_manager = FaultTolerantActorManager(
            _replay_actors,
            max_remote_requests_in_flight_per_actor=(
                self.config.max_requests_in_flight_per_replay_worker
            ),
        )
        self._replay_req_timeout_s = self.config.timeout_s_replay_manager
        self._sample_req_tiemeout_s = self.config.timeout_s_sampler_manager
        self.learner_thread = LearnerThread(self.workers.local_worker())
        self.learner_thread.start()
        self.steps_since_update = defaultdict(int)
        weights = self.workers.local_worker().get_weights()
        self.curr_learner_weights = ray.put(weights)
        self.curr_num_samples_collected = 0
        self._num_ts_trained_since_last_target_update = 0
