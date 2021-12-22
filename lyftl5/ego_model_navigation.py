from collections import deque
from os import close
from typing import Dict
from l5kit.geometry.transform import transform_point, transform_points
import numpy as np
import torch
import torch.nn as nn
from lyftl5.custom_map_api import CustomMapAPI
import random


class EgoModelNavigation(nn.Module):
    def __init__(self, map_api: CustomMapAPI):
        super().__init__()
        self.map_api = map_api
        self.route = None
        self.current_lane_id = None

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        num_of_scenes = len(data_batch['scene_index'])
        
        # If there is no route yet, find one in this scene by taking the ego's current position, and the position
        # of the ego at the end of the scene, and calculate the shortest route between them.
        if self.route is None:
            self.route = []
            self.current_lane_id = []
            for scene_idx in range(num_of_scenes):
                start_position = data_batch["centroid"][scene_idx].cpu().numpy()
                num_simulation_steps = len(data_batch["target_positions"][scene_idx])
                
                # The last target position is the end position.
                end_position_idx = num_simulation_steps - 2
                end_position = data_batch["target_positions"][scene_idx][end_position_idx].cpu().numpy()
                world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
                
                # Transform the end position to the world's reference system.
                end_position = transform_point(end_position, world_from_ego)
                
                # Determine the shortest route between the start and end position.
                route = self.map_api.get_shortest_route(start_position, end_position)
                
                # Pop the first lane of the route queue, which will be the ego's current lane.
                current_lane_id = route.popleft()
                self.route.append(route)
                self.current_lane_id.append(current_lane_id)
        
        steer = torch.zeros([num_of_scenes], dtype=torch.float64)
        for scene_idx in range(num_of_scenes):
            ego_position = data_batch["centroid"][scene_idx].cpu().numpy()
            ego_from_world = data_batch["agent_from_world"][scene_idx].cpu().numpy()
            
            route = self.route[scene_idx]

            # Get the closest lane midpoints for the ego's current lane.
            current_lane_id = self.current_lane_id[scene_idx]
            current_lane_closest_midpoints = self.map_api.get_closest_lane_midpoints(ego_position, current_lane_id)
            current_lane_closest_midpoints = transform_points(current_lane_closest_midpoints, ego_from_world)
            current_lane_closest_midpoints = current_lane_closest_midpoints[current_lane_closest_midpoints[:,0] > 0]

            # Take the next lane of the route, or choose a random next lane that is connected to the current lane, if
            # the current lane is the last lane in the route.
            if len(route) > 0:
                next_lane_id = route[0]  # Peek, don't pop.
            else:
                ahead_lanes_ids = self.map_api.get_ahead_lanes_ids(current_lane_id)
                next_lane_id = random.choice(ahead_lanes_ids)
                route.appendleft(next_lane_id)

            # Get the closest lane midpoints for the ego's next lane.
            next_lane_closest_midpoints = self.map_api.get_closest_lane_midpoints(ego_position, next_lane_id)
            next_lane_closest_midpoints = transform_points(next_lane_closest_midpoints, ego_from_world)
            next_lane_closest_midpoints = next_lane_closest_midpoints[next_lane_closest_midpoints[:,0] > 0]
            
            # If the current lane is closer than the next lane, keep following the current lane's closest midpoint.
            # Otherwise, start following the next lane's closest midpoint.
            if len(current_lane_closest_midpoints) > 0:
                closest_midpoint = current_lane_closest_midpoints[0]
            elif len(next_lane_closest_midpoints) > 0:
                closest_midpoint = next_lane_closest_midpoints[0]
                current_lane_id = route.popleft()
                self.current_lane_id[scene_idx] = current_lane_id
            else:
                current_lane_id = route.popleft()
                self.current_lane_id[scene_idx] = current_lane_id
                continue

            # Steer input is proportional to the y-coordinate of the closest midpoint.
            steer[scene_idx] = closest_midpoint[1]

        data_batch["steer"] = steer
        return data_batch
