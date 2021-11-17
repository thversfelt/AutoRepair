import torch
import torch.nn as nn
from typing import Dict


class EgoModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # In this model, the ego agent will simply stand still.

        positions = torch.tensor([[
            [0.0, 0.0]],  # The relative x,y position of the ego agent to its current centroid in scene 0
            [[0.0, 0.0]   # The relative x,y position of the ego agent to its current centroid in scene 1
             ]])
        yaws = torch.tensor([[
            [0.0]],  # The relative yaw of the ego agent to its current yaw in scene 0
            [[0.0]   # The relative yaw of the ego agent to its current yaw in scene 1
             ]])
        eval_dict = {
            "positions": positions,
            "yaws": yaws
        }

        return eval_dict
