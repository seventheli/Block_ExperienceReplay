import numpy as np
import logging
from gym.spaces import Space
from typing import Dict
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.replay_buffers.utils import SampleBatchType
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import MultiAgentReplayBuffer, ReplayMode
from replay_buffer.replay_node import BaseBuffer
from ray.rllib.utils.replay_buffers import StorageUnit
from ray.util.debug import log_once
from ray.util.timer import _Timer

logger = logging.getLogger(__name__)


class CustomReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            obs_space: Space,
            action_space: Space,
            randomly: bool = False,
            sub_buffer_size: int = 8,
            **kwargs
    ):
        super(CustomReplayBuffer, self).__init__(**kwargs)
        self.base_buffer = BaseBuffer(sub_buffer_size, obs_space, action_space, randomly)

    def sample(self, num_items: int, **kwargs):
        return super(CustomReplayBuffer, self).sample(num_items, **kwargs)

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