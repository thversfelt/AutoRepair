import torch
import torch.nn as nn
from typing import Dict

from lyftl5.custom_map_api import CustomMapAPI
from lyftl5.ego_model_control import EgoModelControl
from lyftl5.ego_model_lane_keeping import EgoModelLaneKeeping
from lyftl5.ego_model_navigation import EgoModelNavigation


class EgoModel(nn.Module):
    def __init__(self, map_api: CustomMapAPI):
        super().__init__()
        self.map_api = map_api
        self.control = EgoModelControl()
        self.lane_keeping = EgoModelLaneKeeping()
        self.navigation = EgoModelNavigation(map_api)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # In this model, the ego agent will simply stand still.

        # lane_keeping_output = self.lane_keeping(data_batch)
        navigation_output = self.navigation.forward(data_batch)
        data_batch["steer"] = torch.Tensor([0.0, 0.0])  # lane_keeping_output["steer"]
        data_batch["acc"] = torch.Tensor([0.3, 0.3])

        # data_batch["steer_acc"] = torch.Tensor([
        #    [0.2, 1.0],  # The steering and acceleration inputs of the ego agent in scene 0.
        #    [0.1, 1.0]  # The steering and acceleration inputs of the ego agent in scene 1.
        # ])

        return self.control.forward(data_batch)
