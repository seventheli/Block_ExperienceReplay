import torch

import torch.nn as nn
from gymnasium.spaces.discrete import Discrete
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomCNN(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space: Discrete, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # self.conv_layers = nn.Sequential(
        #     SlimConv2d(obs_space.shape[-1], 16, kernel=2),
        #     SlimConv2d(16, 32, kernel=2),
        #     SlimConv2d(32, 64, kernel=2),
        # )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(obs_space.shape[-1], 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_space.shape).permute(0, 3, 1, 2)
            conv_out_size = self.conv_layers(dummy_input).flatten(1).shape[-1]

        self.fc_layers = nn.Sequential(
            SlimFC(conv_out_size, 512),
            SlimFC(512, action_space.n)
        )
        self._features = None

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def forward(self, input_dict, state, seq_lens):
        # logging.info(input_dict)
        # logging.info(input_dict["obs"].shape)
        self._features = input_dict["obs"].float()
        # permute b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        self._features = self.conv_layers(self._features)
        self._features = self.fc_layers(self._features.flatten(1))
        return self._features, state

    def value_function(self):
        pass
