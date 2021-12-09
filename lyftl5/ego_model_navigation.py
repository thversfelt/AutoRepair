from typing import Dict
from l5kit.geometry.transform import transform_point
import torch
import torch.nn as nn
from lyftl5.custom_map_api import CustomMapAPI


class EgoModelNavigation(nn.Module):
    def __init__(self, map_api: CustomMapAPI):
        super().__init__()
        self.map_api = map_api
        self.destination = None

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        num_of_scenes = len(data_batch['scene_index'])
        
        if self.destination is None:
            num_of_coordinates = 2  # Number of coordinates (x, y) in a position.
            self.destination = torch.zeros([num_of_scenes, num_of_coordinates], dtype=torch.float64)
            for scene_idx in range(num_of_scenes):
                num_simulation_steps = len(data_batch["target_positions"][scene_idx])
                destination_position_idx = num_simulation_steps - 2  # The last target position is the destination.
                self.destination[scene_idx] = data_batch["target_positions"][scene_idx][destination_position_idx]
                
        steer = torch.zeros([num_of_scenes], dtype=torch.float64)
        for scene_idx in range(num_of_scenes):
            ego_position = data_batch["centroid"][scene_idx].cpu().numpy()
            ego_from_world = data_batch["agent_from_world"][scene_idx].cpu().numpy()
            
            closest_lane_id = self.map_api.get_closest_lane(ego_position)
            closest_midpoints = self.map_api.get_closest_lane_midpoints(ego_position, closest_lane_id)
            closest_midpoint = closest_midpoints[0]
            closest_midpoint = transform_point(closest_midpoint, ego_from_world)  # Transform the closest midpoint to the ego's reference system.
            
            steer[scene_idx] = closest_midpoint[1]  # Steer input is proportional to the relative-y coordinate of the closest midpoint.
        
        eval_dict = {"steer": steer}
        return eval_dict
