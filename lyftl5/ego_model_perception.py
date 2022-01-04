import enum
from l5kit.geometry.transform import transform_point, transform_points
import torch
import torch.nn as nn
from typing import Dict

from lyftl5.custom_map_api import CustomMapAPI
from lyftl5.ego_model_adaptive_cruise_control import EgoModelAdaptiveCruiseControl
from lyftl5.ego_model_control import EgoModelControl
from lyftl5.ego_model_navigation import EgoModelNavigation


class EgoModelPerception(nn.Module):
    def __init__(self, map_api: CustomMapAPI):
        super().__init__()
        self.map_api = map_api
        
        self.ego_route = None  # For each scene id, list of lane id's.
        self.ego_lane = None
        self.ego_speed = None
        
        self.agents_route = None  # For each scene id, dictionary of agent id's with their corresponding routes (lists of lane id's).
        self.agents_lane = None
        self.agents_speed = None

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.update_routes(data_batch)
        self.update_lanes(data_batch)
        self.update_speeds(data_batch)

    def update_routes(self, data_batch: Dict[str, torch.Tensor]):
        num_of_scenes = len(data_batch['scene_index'])

        # EGO
        if self.ego_route is None:
            self.ego_route = [None] * num_of_scenes
            
        for scene_idx in range(num_of_scenes):
            # Ensure the ego route has not been determined for this scene.
            if self.ego_route[scene_idx] is not None:
                continue
            
            # Get the ego reference system to world reference system transformation matrix.
            world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
            
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

        # AGENTS
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

    def update_lanes(self, data_batch: Dict[str, torch.Tensor]):
        num_of_scenes = len(data_batch['scene_index'])
        
        # EGO
        self.ego_lane = []
        for scene_idx in range(num_of_scenes):
            # Get the ego reference system to world reference system transformation matrix.
            world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
            
            # Get the ego's current position.
            position = data_batch["history_positions"][scene_idx][0].cpu().numpy()
            
            # Transform the position to the world reference system.
            position = transform_point(position, world_from_ego)
            
            # The lane of the route that contains the ego in its bounds is the ego's current lane.
            for lane_id in self.ego_route[scene_idx]:
                lane_bounds = self.map_api.get_lane_bounds(lane_id)
                if not self.map_api.in_bounds(position, lane_bounds):
                    continue
                self.ego_lane.append(lane_id)
                break
            
        # AGENTS
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
        pass