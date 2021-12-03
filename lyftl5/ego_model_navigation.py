import math
from typing import Dict
import torch
import torch.nn as nn
from lyftl5.custom_map_api import CustomMapAPI


class EgoModelNavigation(nn.Module):
    def __init__(self, map_api: CustomMapAPI):
        super().__init__()
        self.map_api = map_api

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        num_of_scenes = len(data_batch['scene_index'])
        
        steer = torch.zeros([num_of_scenes], dtype=torch.float64)
        for scene_idx in range(num_of_scenes):
            ego_position = data_batch["centroid"][scene_idx].cpu().numpy()
            ego_from_world = data_batch["agent_from_world"][scene_idx].cpu().numpy()
            
            closest_lane_id = self.map_api.get_closest_lane(ego_position)
            closest_midpoint = self.map_api.get_closest_lane_midpoint(closest_lane_id, ego_from_world)
            while closest_midpoint is None:
                next_lane_id = self.map_api.get_next_lane(closest_lane_id)
                closest_midpoint = self.map_api.get_closest_lane_midpoint(next_lane_id, ego_from_world)
            
            steer[scene_idx] = closest_midpoint[1]
        
        eval_dict = {"steer": steer}
        return eval_dict
