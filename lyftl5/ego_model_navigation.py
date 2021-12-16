from typing import Dict
from l5kit.geometry.transform import transform_point
import numpy as np
import torch
import torch.nn as nn
from lyftl5.custom_map_api import CustomMapAPI


class EgoModelNavigation(nn.Module):
    def __init__(self, map_api: CustomMapAPI):
        super().__init__()
        self.map_api = map_api
        self.route = None
        self.current_lane_idx = None

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        num_of_scenes = len(data_batch['scene_index'])
        
        if self.route is None:
            self.route = []
            self.current_lane_idx = []
            for scene_idx in range(num_of_scenes):
                start_position = data_batch["centroid"][scene_idx].cpu().numpy()
                num_simulation_steps = len(data_batch["target_positions"][scene_idx])
                end_position_idx = num_simulation_steps - 2  # The last target position is the end position.
                end_position = data_batch["target_positions"][scene_idx][end_position_idx].cpu().numpy()
                world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()  # Transform the end position to the world's reference system.
                end_position = transform_point(end_position, world_from_ego)
                route = self.map_api.get_shortest_route(start_position, end_position)
                self.route.append(route)
                self.current_lane_idx.append(0)
        
        steer = torch.zeros([num_of_scenes], dtype=torch.float64)
        for scene_idx in range(num_of_scenes):
            ego_position = data_batch["centroid"][scene_idx].cpu().numpy()
            ego_from_world = data_batch["agent_from_world"][scene_idx].cpu().numpy()
            
            route = self.route[scene_idx]

            current_lane_idx = self.current_lane_idx[scene_idx]
            current_lane_id = route[current_lane_idx]
            current_lane_closest_midpoints = self.map_api.get_closest_lane_midpoints(ego_position, current_lane_id)
            current_lane_closest_midpoint = current_lane_closest_midpoints[0]
            # Transform the closest midpoint to the ego's reference system.
            current_lane_closest_midpoint = transform_point(current_lane_closest_midpoint, ego_from_world)
            current_lane_closest_midpoint_distance = np.linalg.norm(current_lane_closest_midpoint)

            next_lane_idx = min(current_lane_idx + 1, len(route) - 1)
            next_lane_id = route[next_lane_idx]
            next_lane_closest_midpoints = self.map_api.get_closest_lane_midpoints(ego_position, next_lane_id)
            next_lane_closest_midpoint = next_lane_closest_midpoints[0]
            next_lane_closest_midpoint = transform_point(next_lane_closest_midpoint, ego_from_world)  
            next_lane_closest_midpoint_distance = np.linalg.norm(next_lane_closest_midpoint)
            
            if current_lane_closest_midpoint_distance < next_lane_closest_midpoint_distance:
                closest_midpoint = current_lane_closest_midpoint
            else:
                closest_midpoint = next_lane_closest_midpoint
                self.current_lane_idx[scene_idx] = min(current_lane_idx + 1, len(route) - 1)
            
            steer[scene_idx] = closest_midpoint[1]  # Steer input is proportional to the y coordinate of the closest midpoint.
        
        eval_dict = {"steer": steer}
        return eval_dict
