import random
from time import time
from typing import Dict
import numpy as np
import torch
from l5kit.geometry.transform import transform_point, transform_points
from benchmark.custom_map_api import CustomMapAPI

class Agent:
    
    def __init__(self, map: CustomMapAPI, scene_index: int):
        self.map = map
        self.scene_index = scene_index
        self.position = None
        self.velocity = None
        self.speed = None
        self.length = None
        self.route = None
    
class EgoAgent(Agent):

    def update(self, data_batch: Dict[str, torch.Tensor]):
        self.update_position(data_batch)
        self.update_velocity(data_batch)
        self.update_length(data_batch)
        self.update_route(data_batch)
        
    def update_position(self, data_batch: Dict[str, torch.Tensor]):
        # Transform the local position to the world reference system.
        self.position = data_batch["centroid"][self.scene_index].cpu().numpy()
    
    def update_velocity(self, data_batch: Dict[str, torch.Tensor]):   
        # Get the availability of the ego in the scene's frames.
        availability = data_batch["history_availabilities"][self.scene_index].cpu().numpy()
            
        # Ensure the agent's historical positions are known (are available).
        if not all(availability):
            self.speed = data_batch['speed'][self.scene_index].cpu().numpy()
            return

        # Get the ego reference system to world reference system transformation matrix.
        world_from_ego = data_batch["world_from_agent"][self.scene_index].cpu().numpy()
        
        # Get the previous position of the agent in the scene.
        previous_local_position = data_batch["history_positions"][self.scene_index][1].cpu().numpy()
        
        # Transform the position to the world reference system.
        previous_position = transform_point(previous_local_position, world_from_ego)
        
        # Calculate the agent's velocity.
        timestep = 0.1
        self.velocity = (self.position - previous_position) / timestep
        
        # Calculate the agent's speed.
        self.speed = np.linalg.norm(self.velocity)
    
    def update_length(self, data_batch: Dict[str, torch.Tensor]):
        self.length = data_batch['extent'][self.scene_index][0].cpu().numpy()
        
    def update_route(self, data_batch: Dict[str, torch.Tensor]):
        if self.route is None:
            self.determine_route(data_batch)
        else:
            self.adjust_route()
    
    def determine_route(self, data_batch: Dict[str, torch.Tensor]):
        # Get the ego reference system to world reference system transformation matrix.
        world_from_ego = data_batch["world_from_agent"][self.scene_index].cpu().numpy()
        
        # Get the availability of the ego in the scene's frames.
        availability = data_batch["target_availabilities"][self.scene_index].cpu().numpy()
        
        # Get the trajectory of the ego in the scene.
        trajectory = data_batch["target_positions"][self.scene_index].cpu().numpy()
        
        # Filter the trajectory based on the ego's availability for each frame.
        trajectory = trajectory[availability]
        
        # Transform the trajectory to the world reference system.
        trajectory = transform_points(trajectory, world_from_ego)
        
        # Get the route that matches the trajectory.
        self.route = self.map.get_route(trajectory)
        
    def adjust_route(self):
        # Has next lane.
        has_next_lane = len(self.route) >= 2
        
        # Ensure there are at least two lanes (the current lane, and the next lane), otherwise extend the route.
        if not has_next_lane:
            return self.extend_route()
        
        # Get the next lane's id.
        next_lane_id = self.route[1]
        
        # Check if the ego is in its next lane. 
        in_next_lane = self.map.in_lane_bounds(self.position, next_lane_id)
        
        # The first lane of the route is not the current lane anymore, remove it from the route.
        if in_next_lane:
            self.route.popleft()
            
    def extend_route(self):
        current_lane_id = self.route[0]
        ahead_lanes_ids = self.map.get_ahead_lanes_ids(current_lane_id)
        
        if len(ahead_lanes_ids) == 0:
            return
        
        next_lane_id = random.choice(ahead_lanes_ids)
        self.route.append(next_lane_id)

class VehicleAgent(Agent):
    
    def __init__(self, map: CustomMapAPI, scene_index: int, index: int, id: int):
        super().__init__(map, scene_index)
        self.index = index
        self.id = id

    def update(self, data_batch: Dict[str, torch.Tensor]):
        self.update_position(data_batch)
        self.update_velocity(data_batch)
        
    def update_position(self, data_batch: Dict[str, torch.Tensor]):
        # Get the ego reference system to world reference system transformation matrix.
        world_from_ego = data_batch["world_from_agent"][self.scene_index].cpu().numpy()
        
        # Get the agent's local position.
        self.local_position = data_batch["all_other_agents_history_positions"][self.scene_index][self.index][0].cpu().numpy()

        # Transform the position to the world reference system.
        self.position = transform_point(self.local_position, world_from_ego)
    
    def update_velocity(self, data_batch: Dict[str, torch.Tensor]):
        # Get the availability of the agent in the scene's frames.
        availability = data_batch["all_other_agents_history_availability"][self.scene_index][self.index].cpu().numpy()

        # Ensure the agent's historical positions are known (are available).
        if not all(availability):
            return

        # Get the ego reference system to world reference system transformation matrix.
        world_from_ego = data_batch["world_from_agent"][self.scene_index].cpu().numpy()
        
        # Get the previous position of the agent in the scene.
        previous_local_position = data_batch["all_other_agents_history_positions"][self.scene_index][self.index][1].cpu().numpy()
        
        # Transform the position to the world reference system.
        previous_position = transform_point(previous_local_position, world_from_ego)
        
        # Calculate the agent's velocity.
        timestep = 0.1
        self.velocity = (self.position - previous_position) / timestep
        
        # Calculate the agent's speed.
        self.speed = np.linalg.norm(self.velocity)
    
    def update_velocity(self, data_batch: Dict[str, torch.Tensor]):
        # Get the availability of the agent in the scene's frames.
        availability = data_batch["all_other_agents_history_availability"][self.scene_index][self.index].cpu().numpy()

        # Ensure the agent's historical positions are known (are available).
        if not all(availability):
            return

        # Get the ego reference system to world reference system transformation matrix.
        world_from_ego = data_batch["world_from_agent"][self.scene_index].cpu().numpy()
        
        # Get the previous position of the agent in the scene.
        previous_local_position = data_batch["all_other_agents_history_positions"][self.scene_index][self.index][1].cpu().numpy()
        
        # Transform the position to the world reference system.
        previous_position = transform_point(previous_local_position, world_from_ego)
        
        # Calculate the agent's velocity.
        timestep = 0.1
        self.velocity = (self.position - previous_position) / timestep
        
        # Calculate the agent's speed.
        self.speed = np.linalg.norm(self.velocity)