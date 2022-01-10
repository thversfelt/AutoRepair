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
            
            # Get the closest lane midpoints for the ego's current lane.
            current_lane_id = self.perception.ego_route[scene_idx][0]
            current_lane_closest_midpoints = self.perception.map_api.get_closest_lane_midpoints(ego_position, current_lane_id)
            current_lane_closest_midpoints = transform_points(current_lane_closest_midpoints, ego_from_world)
            current_lane_closest_midpoints = current_lane_closest_midpoints[current_lane_closest_midpoints[:,0] > 0]
        
            # Get the closest lane midpoints for the ego's next lane.
            next_lane_id = self.perception.ego_route[scene_idx][1]
            next_lane_closest_midpoints = self.perception.map_api.get_closest_lane_midpoints(ego_position, next_lane_id)
            next_lane_closest_midpoints = transform_points(next_lane_closest_midpoints, ego_from_world)
            next_lane_closest_midpoints = next_lane_closest_midpoints[next_lane_closest_midpoints[:,0] > 0]
        
            if len(current_lane_closest_midpoints) > 0:
                closest_midpoint = current_lane_closest_midpoints[0]
            elif len(next_lane_closest_midpoints) > 0:
                closest_midpoint = next_lane_closest_midpoints[0]
        
            # Steer input is proportional to the y-coordinate of the closest midpoint.
            steer[scene_idx] = closest_midpoint[1]
        
        data_batch["steer"] = steer
        return data_batch
