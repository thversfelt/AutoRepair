import math

import torch
import torch.nn as nn
from typing import Dict
import numpy as np

from lyftl5.physics_model import PhysicsModel


class EgoModel(nn.Module):
    def __init__(self, physics_model: PhysicsModel):
        super().__init__()
        self.physics_model = physics_model

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # In this model, the ego agent will simply stand still.

        data_batch["steer_acc"] = torch.Tensor([
            [0.2, 1.0],  # The steering and acceleration inputs of the ego agent in scene 0.
            [0.1, 1.0]  # The steering and acceleration inputs of the ego agent in scene 1.
        ])

        return self.physics_model.forward(data_batch)
