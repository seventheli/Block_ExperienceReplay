import numpy as np
import logging
from gym.spaces import Space
from typing import Dict
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.replay_buffers.utils import SampleBatchType
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import MultiAgentReplayBuffer, ReplayMode
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from replay_buffer.replay_node import BaseBuffer
from ray.rllib.utils.replay_buffers import StorageUnit
from ray.util.debug import log_once
from ray.util.timer import _Timer

logger = logging.getLogger(__name__)


class BlockedPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            randomly: bool = False,
            sub_buffer_size: int = 32,
            beta=0.6,
            **kwargs
    ):
        super(BlockedPrioritizedReplayBuffer, self).__init__(**kwargs)
        self.beta = beta
        self.base_buffer = BaseBuffer(sub_buffer_size, obs_space, action_space, randomly)

    def sample(self, num_items: int, **kwargs):
        return super(BlockedPrioritizedReplayBuffer, self).sample(num_items, **kwargs)

    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Adds a batch of experiences to this buffer.

        Splits batch into chunks of timesteps, sequences or episodes, depending on
        `self._storage_unit`. Calls `self._add_single_batch` to add resulting slices
        to the buffer storage.

        Args:
            batch: Batch to add.
            ``**kwargs``: Forward compatibility kwargs.
        """
        if not batch.count > 0:
            return
        buffer = self.base_buffer

        self.base_buffer.add(batch)
        if buffer.full:
            data = buffer.sample()
            weight = np.mean(data.get("weights"))
            buffer.reset()
            self._add_single_batch(data, weight=weight)


@DeveloperAPI
class MultiAgentBlockedPrioritizedReplayBuffer(MultiAgentReplayBuffer, BlockedPrioritizedReplayBuffer):
    """A prioritized replay buffer shard for multiagent setups.

    This buffer is meant to be run in parallel to distribute experiences
    across `num_shards` shards. Unlike simpler buffers, it holds a set of
    buffers - one for each policy ID.
    """

    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            sub_buffer_size: int = 1,
            capacity: int = 10000,
            num_shards: int = 1,
            learning_starts: int = 1000,
            rollout_fragment_length: int = 4,
            replay_mode: str = "independent",
            replay_sequence_override: bool = True,
            replay_sequence_length: int = 1,
            replay_burn_in: int = 0,
            replay_zero_init_states: bool = True,
            prioritized_replay_alpha: float = 0.6,
            prioritized_replay_beta: float = 0.4,
            prioritized_replay_eps: float = 1e-6,
            **kwargs
    ):
        """Initializes a MultiAgentReplayBuffer instance.

        Args:
            capacity: The capacity of the buffer, measured in `storage_unit`.
            storage_unit: Either 'timesteps', 'sequences' or
                'episodes'. Specifies how experiences are stored. If they
                are stored in episodes, replay_sequence_length is ignored.
                If they are stored in episodes, replay_sequence_length is
                ignored.
            num_shards: The number of buffer shards that exist in total
                (including this one).
            learning_starts: Number of timesteps after which a call to
                `replay()` will yield samples (before that, `replay()` will
                return None).
            replay_mode: One of "independent" or "lockstep". Determines,
                whether batches are sampled independently or to an equal
                amount.
            replay_sequence_override: If True, ignore sequences found in incoming
                batches, slicing them into sequences as specified by
                `replay_sequence_length` and `replay_sequence_burn_in`. This only has
                an effect if storage_unit is `sequences`.
            replay_sequence_length: The sequence length (T) of a single
                sample. If > 1, we will sample B x T from this buffer.
            replay_burn_in: The burn-in length in case
                `replay_sequence_length` > 0. This is the number of timesteps
                each sequence overlaps with the previous one to generate a
                better internal state (=state after the burn-in), instead of
                starting from 0.0 each RNN rollout.
            replay_zero_init_states: Whether the initial states in the
                buffer (if replay_sequence_length > 0) are alwayas 0.0 or
                should be updated with the previous train_batch state outputs.
            underlying_buffer_config: A config that contains all necessary
                constructor arguments and arguments for methods to call on
                the underlying buffers. This replaces the standard behaviour
                of the underlying PrioritizedReplayBuffer. The config
                follows the conventions of the general
                replay_buffer_config. kwargs for subsequent calls of methods
                may also be included. Example:
                "replay_buffer_config": {"type": PrioritizedReplayBuffer,
                "capacity": 10, "storage_unit": "timesteps",
                prioritized_replay_alpha: 0.5, prioritized_replay_beta: 0.5,
                prioritized_replay_eps: 0.5}
            prioritized_replay_alpha: Alpha parameter for a prioritized
                replay buffer. Use 0.0 for no prioritization.
            prioritized_replay_beta: Beta parameter for a prioritized
                replay buffer.
            prioritized_replay_eps: Epsilon parameter for a prioritized
                replay buffer.
            ``**kwargs``: Forward compatibility kwargs.
        """

        if "replay_mode" in kwargs and (
                kwargs["replay_mode"] == "lockstep"
                or kwargs["replay_mode"] == ReplayMode.LOCKSTEP
        ):
            if log_once("lockstep_mode_not_supported"):
                logger.error(
                    "Replay mode `lockstep` is not supported for "
                    "MultiAgentPrioritizedReplayBuffer. "
                    "This buffer will run in `independent` mode."
                )
            kwargs["replay_mode"] = "independent"
        prioritized_replay_buffer_config = {
            "type": BlockedPrioritizedReplayBuffer,
            "action_space": action_space,
            "obs_space": obs_space,
            "storage_unit": StorageUnit.FRAGMENTS,
            "randomly": False,
            "sub_buffer_size": sub_buffer_size,
            "alpha": prioritized_replay_alpha,
            "beta": prioritized_replay_beta
        }
        MultiAgentReplayBuffer.__init__(
            self,
            capacity=capacity,
            num_shards=num_shards,
            storage_unit=StorageUnit.FRAGMENTS,
            replay_sequence_override=replay_sequence_override,
            learning_starts=learning_starts,
            replay_mode=replay_mode,
            replay_sequence_length=replay_sequence_length,
            replay_burn_in=replay_burn_in,
            replay_zero_init_states=replay_zero_init_states,
            underlying_buffer_config=prioritized_replay_buffer_config,
            **kwargs,
        )
        self.rollout_fragment_length = rollout_fragment_length
        self.prioritized_replay_eps = prioritized_replay_eps
        self.update_priorities_timer = _Timer()

    @DeveloperAPI
    @override(ReplayBuffer)
    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Adds a batch to the appropriate policy's replay buffer.

        Turns the batch into a MultiAgentBatch of the DEFAULT_POLICY_ID if
        it is not a MultiAgentBatch. Subsequently, adds the individual policy
        batches to the storage.

        Args:
            batch : The batch to be added.
            ``**kwargs``: Forward compatibility kwargs.
        """
        if batch is None:
            if log_once("empty_batch_added_to_buffer"):
                logger.info(
                    "A batch that is `None` was added to {}. This can be "
                    "normal at the beginning of execution but might "
                    "indicate an issue.".format(type(self).__name__)
                )
            return
        # Make a copy so the replay buffer doesn't pin plasma memory.
        batch = batch.copy()
        # Handle everything as if multi-agent.
        batch = batch.as_multi_agent()
        with self.add_batch_timer:
            pids_and_batches = self._maybe_split_into_policy_batches(batch)
            for policy_id, sample_batch in pids_and_batches.items():
                for s_batch in sample_batch.timeslices(self.rollout_fragment_length):
                    self._add_to_underlying_buffer(policy_id, s_batch)

        self._num_added += batch.count

    @DeveloperAPI
    @override(PrioritizedReplayBuffer)
    def update_priorities(self, prio_dict: Dict) -> None:
        """Updates the priorities of underlying replay buffers.

        Computes new priorities from td_errors and prioritized_replay_eps.
        These priorities are used to update underlying replay buffers per
        policy_id.

        Args:
            prio_dict: A dictionary containing td_errors for
            batches saved in underlying replay buffers.
        """
        with self.update_priorities_timer:
            for policy_id, (batch_indexes, td_errors) in prio_dict.items():
                new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                self.replay_buffers[policy_id].update_priorities(
                    batch_indexes, new_priorities
                )
