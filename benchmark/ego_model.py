import numpy as np
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
    def __init__(self, map: CustomMapAPI):
        super().__init__()
        self.perception = EgoModelPerception(map)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.perception.process(data_batch)
        num_of_scenes = len(self.perception.scenes)
        
        # The x,y positions and yaws of the ego agent's reference system in each scene.
        positions = np.zeros([num_of_scenes, 2])
        yaws = np.zeros([num_of_scenes, 1])
    
        for _, scene in self.perception.scenes.items():
            control = EgoModelControl()
            navigation = EgoModelNavigation()
            
            steer = navigation.process(scene)
            acc = 1.0
            
            position, yaw = control.process(scene.ego, steer, acc)
            positions[scene.index] = position
            yaws[scene.index] = yaw

        positions = torch.from_numpy(positions)
        yaws = torch.from_numpy(yaws)
        
        return {
            "positions": torch.reshape(positions, [num_of_scenes, 1, 2]),
            "yaws": torch.reshape(yaws, [num_of_scenes, 1, 1])
        }
