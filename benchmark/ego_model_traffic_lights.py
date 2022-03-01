from typing import Dict, List
from l5kit.geometry.transform import transform_point
import numpy as np
import torch
from torch import nn
from benchmark.ego_model_perception import EgoModelPerception


class EgoModelTrafficLights(nn.Module):
    def __init__(self, perception: EgoModelPerception):
        super().__init__()
        self.perception = perception

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        num_of_scenes = len(data_batch['scene_index'])
      
        acc = torch.zeros([num_of_scenes], dtype=torch.float64)
        for scene_idx in range(num_of_scenes):
            current_lane_id = self.perception.ego_route[scene_idx][0]
            has_active_traffic_light = current_lane_id in self.perception.traffic_lights[scene_idx]
            # Get the ego's absolute speed.
            ego_speed = self.perception.ego_speed[scene_idx]
            if has_active_traffic_light and self.perception.traffic_lights[scene_idx][current_lane_id] == 0 and ego_speed >= 0.0:
                acc[scene_idx] = -1.0
            else:
                acc[scene_idx] = 1.0

        data_batch["acc"] = acc
        return data_batch
