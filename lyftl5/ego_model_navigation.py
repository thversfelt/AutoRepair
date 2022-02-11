from collections import deque
from os import close
from typing import Dict
from l5kit.geometry.transform import transform_point, transform_points
import numpy as np
import torch
import torch.nn as nn
from lyftl5.custom_map_api import CustomMapAPI
import random

from lyftl5.ego_model_perception import EgoModelPerception


class EgoModelNavigation(nn.Module):
    def __init__(self, perception: EgoModelPerception):
        super().__init__()
        self.perception = perception

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        num_of_scenes = len(data_batch['scene_index'])
        
        steer = torch.zeros([num_of_scenes], dtype=torch.float64)
        for scene_idx in range(num_of_scenes):
            ego_position = data_batch["centroid"][scene_idx].cpu().numpy()
            ego_from_world = data_batch["agent_from_world"][scene_idx].cpu().numpy()
            
            closest_midpoint = None
            
            for lane_id in self.perception.ego_route[scene_idx]:
                # Get the closest lane midpoints for the ego's current lane.
                lane_closest_midpoints = self.perception.map_api.get_closest_lane_midpoints(ego_position, lane_id)
                lane_closest_midpoints = transform_points(lane_closest_midpoints, ego_from_world)
                lane_closest_midpoints = lane_closest_midpoints[lane_closest_midpoints[:,0] > 0]
                
                if len(lane_closest_midpoints) == 0:
                    continue
                else:
                    closest_midpoint = lane_closest_midpoints[0]
                    break

            # Steer input is proportional to the y-coordinate of the closest midpoint.
            steer[scene_idx] = 0.5 * closest_midpoint[1]
        
        data_batch["steer"] = steer
        return data_batch
