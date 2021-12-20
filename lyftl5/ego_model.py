import torch
import torch.nn as nn
from typing import Dict

from lyftl5.custom_map_api import CustomMapAPI
from lyftl5.ego_model_adaptive_cruise_control import EgoModelAdaptiveCruiseControl
from lyftl5.ego_model_control import EgoModelControl
from lyftl5.ego_model_navigation import EgoModelNavigation


class EgoModel(nn.Module):
    def __init__(self, map_api: CustomMapAPI):
        super().__init__()
        self.map_api = map_api
        self.control = EgoModelControl()
        self.navigation = EgoModelNavigation(map_api)
        self.adaptive_cruise_control = EgoModelAdaptiveCruiseControl(map_api)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        num_of_scenes = len(data_batch['scene_index'])
        
        navigation_output = self.navigation.forward(data_batch)
        adaptive_cruise_control_output = self.adaptive_cruise_control.forward(data_batch)
        
        data_batch["steer"] = navigation_output["steer"]  # torch.Tensor([0.0, 0.0])
        data_batch["acc"] = 0.3 * torch.ones(num_of_scenes)
        
        return self.control.forward(data_batch)
