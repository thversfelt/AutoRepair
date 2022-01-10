import random
from l5kit.geometry.transform import transform_point, transform_points
import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from lyftl5.custom_map_api import CustomMapAPI


class EgoModelPerception(nn.Module):
    def __init__(self, map_api: CustomMapAPI):
        super().__init__()
        self.map_api = map_api
        
        self.ego_route = {}  # For each scene id, list of lane id's.
        self.ego_speed = {}
        
        self.agents_route = {}  # For each scene id, dictionary of agent id's with their corresponding routes (lists of lane id's).
        self.agents_speed = {}

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.update_ego_route(data_batch)
        #self.update_agent_routes(data_batch)
        
        #self.update_speeds(data_batch)

    def update_ego_route(self, data_batch: Dict[str, torch.Tensor]):
        num_of_scenes = len(data_batch['scene_index'])

        for scene_idx in range(num_of_scenes):
            # Get the ego reference system to world reference system transformation matrix.
            world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
            
            if scene_idx not in self.ego_route:
                # TODO: ADD HISTORY POSITIONS AND AVAILABILITIES TO TRAJECTORY.
                
                # Get the availability of the ego in the scene's frames.
                availability = data_batch["target_availabilities"][scene_idx].cpu().numpy()
                
                # Get the trajectory of the ego in the scene.
                trajectory = data_batch["target_positions"][scene_idx].cpu().numpy()
                
                # Filter the trajectory based on the ego's availability for each frame.
                trajectory = trajectory[availability]
                
                # Transform the trajectory to the world reference system.
                trajectory = transform_points(trajectory, world_from_ego)
                
                # Get the route that matches the trajectory.
                route = self.map_api.get_route(trajectory)
                
                # Assign the ego's route in this scene.
                self.ego_route[scene_idx] = route
            else:
                # Get the ego's current position.
                current_position = data_batch["history_positions"][scene_idx][0].cpu().numpy()
                
                # Transform the position to the world reference system.
                current_position = transform_point(current_position, world_from_ego)
                
                # Get the next lane's id.
                next_lane_id = self.ego_route[scene_idx][1]
                
                # Get the next lane's bounds.
                next_lane_bounds = self.map_api.get_lane_bounds(next_lane_id)
                
                # Determine if the ego is in the next lane.
                in_next_lane = self.map_api.in_bounds(current_position, next_lane_bounds)
                
                # The first lane of the route is not the current lane anymore, remove it.
                if in_next_lane:
                    self.ego_route[scene_idx].popleft()
        
            current_lane_id = self.ego_route[scene_idx][0]
            last_lane_id = self.ego_route[scene_idx][-1]
            
            if current_lane_id == last_lane_id:
                ahead_lanes_ids = self.map_api.get_ahead_lanes_ids(current_lane_id)
                next_lane_id = random.choice(ahead_lanes_ids)
                self.ego_route[scene_idx].append(next_lane_id)

    def update_agent_routes(self, data_batch: Dict[str, torch.Tensor]):
        num_of_scenes = len(data_batch['scene_index'])
        
        if self.agents_route is None:
            self.agents_route = [{}] * num_of_scenes
        
        for scene_idx in range(num_of_scenes):
            
            # Get the ego reference system to world reference system transformation matrix.
            world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
            
            # Get the list of agent id's in this scene.
            agents_ids = data_batch["all_other_agents_track_ids"][scene_idx].cpu().numpy()
            
            for agent_idx, agent_id in enumerate(agents_ids):
                # Ensure the agent has a valid id (an id of 0 is invalid).
                if agent_id == 0:
                    continue
                
                # Ensure the agent's route has not been determined for this scene.
                if agent_id in self.agents_route[scene_idx]:
                    continue
                
                # TODO: ADD HISTORY POSITIONS AND AVAILABILITIES TO TRAJECTORY.
                
                # Get the availability of the agent in the scene's frames.
                availability = data_batch["all_other_agents_future_availability"][scene_idx][agent_idx].cpu().numpy()
                
                # Get the trajectory of the agent in the scene.
                trajectory = data_batch["all_other_agents_future_positions"][scene_idx][agent_idx].cpu().numpy()
                
                # Filter the trajectory based on that agent's availability for each frame.
                trajectory = trajectory[availability]
                
                # Transform the trajectory to the world reference system.
                trajectory = transform_points(trajectory, world_from_ego)
                
                # Get the route that matches the trajectory.
                route = self.map_api.get_route(trajectory)
                
                # Assign this agent's route in this scene.
                self.agents_route[scene_idx][agent_id] = route
    
    
    def update_agent_lanes(self, data_batch: Dict[str, torch.Tensor]):
        num_of_scenes = len(data_batch['scene_index'])
        
        self.agents_lane = []
        for scene_idx in range(num_of_scenes):
            self.agents_lane.append({})
                
            # Get the ego reference system to world reference system transformation matrix.
            world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
            
            # Get the list of agent id's in this scene.
            agents_ids = data_batch["all_other_agents_track_ids"][scene_idx].cpu().numpy()
            
            for agent_idx, agent_id in enumerate(agents_ids):
                # Make sure the agent has a valid id (an id of 0 is invalid).
                if agent_id == 0:
                    continue
                
                # Get the position of the agent in the scene.
                position = data_batch["all_other_agents_history_positions"][scene_idx][agent_idx][0].cpu().numpy()
                
                # Transform the position to the world reference system.
                position = transform_point(position, world_from_ego)
                
                # The lane of the route that contains the ego in its bounds is the ego's current lane.
                for lane_id in self.agents_route[scene_idx][agent_id]:
                    lane_bounds = self.map_api.get_lane_bounds(lane_id)
                    if not self.map_api.in_bounds(position, lane_bounds):
                        continue
                    self.agents_lane[scene_idx][agent_id] = lane_id
                    break
    
    
    
    def update_speeds(self, data_batch: Dict[str, torch.Tensor]):
        num_of_scenes = len(data_batch['scene_index'])
        
        # EGO
        self.ego_speed = []
        for scene_idx in range(num_of_scenes):
            
            # Get the availability of the agent in the scene's frames.
            availability = data_batch["history_availabilities"][scene_idx].cpu().numpy()
            
            if not all(availability):
                speed = data_batch['speed'][scene_idx]
                self.ego_speed.append(speed)
                continue
            
            # Get the ego reference system to world reference system transformation matrix.
            world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
            
            # Get the position of the ego in the scene.
            position = data_batch["history_positions"][scene_idx][0].cpu().numpy()
            
            # Transform the position to the world reference system.
            position = transform_point(position, world_from_ego)
            
            # Get the previous position of the ego in the scene.
            previous_position = data_batch["history_positions"][scene_idx][1].cpu().numpy()
            
            # Transform the position to the world reference system.
            previous_position = transform_point(previous_position, world_from_ego)
            
            # Calculate the ego's speed.
            speed = np.linalg.norm(position - previous_position)
            
            # Set the ego's speed for this scene.
            self.ego_speed.append(speed)
            
        # AGENTS
        self.agents_speed = []
        for scene_idx in range(num_of_scenes):
            self.agents_speed.append({})
            
            # Get the list of agent id's in this scene.
            agents_ids = data_batch["all_other_agents_track_ids"][scene_idx].cpu().numpy()
            
            for agent_idx, agent_id in enumerate(agents_ids):
                # Make sure the agent has a valid id (an id of 0 is invalid).
                if agent_id == 0:
                    continue
                
                # Get the availability of the agent in the scene's frames.
                availability = data_batch["all_other_agents_history_availability"][scene_idx][agent_idx].cpu().numpy()
                
                if not all(availability):
                    self.agents_speed[scene_idx][agent_id] = 0.0
                    continue
                
                # Get the ego reference system to world reference system transformation matrix.
                world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
                
                # Get the position of the agent in the scene.
                position = data_batch["all_other_agents_history_positions"][scene_idx][agent_idx][0].cpu().numpy()
                
                # Transform the position to the world reference system.
                position = transform_point(position, world_from_ego)
                
                # Get the previous position of the agent in the scene.
                previous_position = data_batch["all_other_agents_history_positions"][scene_idx][agent_idx][1].cpu().numpy()
                
                # Transform the position to the world reference system.
                previous_position = transform_point(previous_position, world_from_ego)
                
                # Calculate the agent's speed.
                speed =  np.linalg.norm(position - previous_position)
                
                # Set the agent's speed for this scene.
                self.agents_speed[scene_idx][agent_id] = speed
