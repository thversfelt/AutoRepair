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
            #traffic_lights_ids = data_batch["traffic_lights_ids"].cpu().numpy()
            #traffic_lights_colors = data_batch["traffic_lights_colors"].cpu().numpy()
            
            for lane_id in self.perception.ego_route[scene_idx]:
                traffic_control_ids = self.perception.map_api.get_lane_traffic_control_ids(lane_id)
                for traffic_control_id in traffic_control_ids:
                    break
                    
            
            acc[scene_idx] = 1.0

        data_batch["acc"] = acc
        return data_batch
