import logging
import numpy as np
from gym.spaces import Space
from ray.rllib.utils.replay_buffers.utils import SampleBatchType
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from replay_buffer.replay_node import BaseBuffer
from ray.rllib.utils.replay_buffers import StorageUnit

logger = logging.getLogger(__name__)


class CustomPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            randomly: bool = False,
            capacity: int = 1000000,
            sub_buffer_size: int = 32,
            prioritized_replay_alpha: float = 0.6,
            **kwargs
    ):
        super(CustomPrioritizedReplayBuffer, self).__init__(capacity=capacity,
                                                            storage_unit=StorageUnit.FRAGMENTS,
                                                            alpha=prioritized_replay_alpha, **kwargs)
        self.base_buffer = BaseBuffer(sub_buffer_size, obs_space, action_space, randomly)

    def sample(self, num_items: int, **kwargs):
        return super(CustomPrioritizedReplayBuffer, self).sample(num_items, **kwargs)

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
