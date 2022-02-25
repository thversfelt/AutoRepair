import torch
import torch.nn as nn
from typing import Dict

from benchmark.custom_map_api import CustomMapAPI
from benchmark.ego_model_adaptive_cruise_control import EgoModelAdaptiveCruiseControl
from benchmark.ego_model_control import EgoModelControl
from benchmark.ego_model_navigation import EgoModelNavigation
from benchmark.ego_model_perception import EgoModelPerception
from benchmark.ego_model_traffic_lights import EgoModelTrafficLights


class EgoModel(nn.Module):
    def __init__(self, map_api: CustomMapAPI):
        super().__init__()
        self.perception = EgoModelPerception(map_api)
        self.control = EgoModelControl(self.perception)
        self.navigation = EgoModelNavigation(self.perception)
        self.adaptive_cruise_control = EgoModelAdaptiveCruiseControl(self.perception)
        self.traffic_lights = EgoModelTrafficLights(self.perception)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.perception.forward(data_batch)
        
        self.navigation.forward(data_batch)
        #self.adaptive_cruise_control.forward(data_batch)
        self.traffic_lights.forward(data_batch)
        
        return self.control.forward(data_batch)
