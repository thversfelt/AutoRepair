import random
from turtle import pos
from l5kit.geometry.transform import transform_point, transform_points
import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from benchmark.custom_map_api import CustomMapAPI


class EgoModelPerception(nn.Module):
    def __init__(self, map_api: CustomMapAPI):
        super().__init__()
        self.map_api = map_api
        
        self.timestep: float = 0.1  # [s]
        
        self.ego_local_position = {}
        self.ego_position = {}
        self.ego_route = {}  # For each scene id, list of lane id's.
        self.ego_speed = {}
        self.ego_leader = {}
        self.ego_leader_distance = {}
        
        self.agents_local_position = {}
        self.agents_position = {}
        self.agents_route = {}  # For each scene id, dictionary of agent id's with their corresponding routes (lists of lane id's).
        self.agents_speed = {}
        
        self.traffic_lights = {} # traffic_lights[scene_id][lane_id] = GREEN, YELLOW, or RED

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.update_ego_position(data_batch)
        self.update_agents_position(data_batch)
        
        self.update_ego_route(data_batch)
        self.update_agents_route(data_batch)
        
        self.update_ego_speed(data_batch)
        self.update_agents_speed(data_batch)
        
        self.update_ego_leader(data_batch)
        self.update_traffic_lights(data_batch)

    def update_ego_position(self, data_batch: Dict[str, torch.Tensor]):
        num_of_scenes = len(data_batch['scene_index'])

        for scene_idx in range(num_of_scenes):
            # Get the ego reference system to world reference system transformation matrix.
            world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
            
            # Get the ego's local position.
            local_position = data_batch["history_positions"][scene_idx][0].cpu().numpy()
            
            # Assign the ego's position in this scene.
            self.ego_local_position[scene_idx] = local_position
            
            # Transform the local position to the world reference system.
            position = transform_point(local_position, world_from_ego)
            
            # Assign the ego's position in this scene.
            self.ego_position[scene_idx] = position

    def update_agents_position(self, data_batch: Dict[str, torch.Tensor]):
        num_of_scenes = len(data_batch['scene_index'])
        
        for scene_idx in range(num_of_scenes):
            # Get the ego reference system to world reference system transformation matrix.
            world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
            
            # Initialize the agents local position dictionary for each scene.
            if scene_idx not in self.agents_local_position:
                self.agents_local_position[scene_idx] = {}
            
            # Initialize the agents position dictionary for each scene.
            if scene_idx not in self.agents_position:
                self.agents_position[scene_idx] = {}
            
            # Get the list of agent id's in this scene.
            agents_ids = data_batch["all_other_agents_track_ids"][scene_idx].cpu().numpy()
            
            for agent_idx, agent_id in enumerate(agents_ids):
                # Ensure the agent has a valid id (an id of 0 is invalid).
                if agent_id == 0:
                    continue
                
                # Get the agent's local position.
                local_position = data_batch["all_other_agents_history_positions"][scene_idx][agent_idx][0].cpu().numpy()
                
                # Assign the agent's local position in this scene.
                self.agents_local_position[scene_idx][agent_id] = local_position
                
                # Transform the position to the world reference system.
                position = transform_point(local_position, world_from_ego)
                
                # Assign the agent's position in this scene.
                self.agents_position[scene_idx][agent_id] = position

    def update_ego_route(self, data_batch: Dict[str, torch.Tensor]):
        num_of_scenes = len(data_batch['scene_index'])

        for scene_idx in range(num_of_scenes):
            # Get the ego reference system to world reference system transformation matrix.
            world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
            
            if scene_idx not in self.ego_route:
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
                # Has next lane.
                has_next_lane = len(self.ego_route[scene_idx]) > 1
                
                if not has_next_lane:
                    continue
                
                # Get the ego's current position.
                position = self.ego_position[scene_idx]
                
                # Get the next lane's id.
                next_lane_id = self.ego_route[scene_idx][1]
                
                # Check if the ego is in its next lane. 
                in_next_lane = self.map_api.in_lane_bounds(position, next_lane_id)
                
                # The first lane of the route is not the current lane anymore, remove it.
                if in_next_lane:
                    self.ego_route[scene_idx].popleft()

            if len(self.ego_route[scene_idx]) == 1:
                current_lane_id = self.ego_route[scene_idx][0]
                ahead_lanes_ids = self.map_api.get_ahead_lanes_ids(current_lane_id)
                if len(ahead_lanes_ids) > 0:
                    next_lane_id = random.choice(ahead_lanes_ids)
                    self.ego_route[scene_idx].append(next_lane_id)

    def update_agents_route(self, data_batch: Dict[str, torch.Tensor]):
        num_of_scenes = len(data_batch['scene_index'])
        
        for scene_idx in range(num_of_scenes):
            # Get the ego reference system to world reference system transformation matrix.
            world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
            
            # Initialize the agent route dictionary for each scene.
            if scene_idx not in self.agents_route:
                self.agents_route[scene_idx] = {}
            
            # Get the list of agent id's in this scene.
            agents_ids = data_batch["all_other_agents_track_ids"][scene_idx].cpu().numpy()
            
            for agent_idx, agent_id in enumerate(agents_ids):
                # Ensure the agent has a valid id (an id of 0 is invalid).
                if agent_id == 0:
                    continue
                
                if agent_id not in self.agents_route[scene_idx]:
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
                else:
                    # Has a route.
                    has_route = self.agents_route[scene_idx][agent_id] != None
                    
                    if not has_route:
                        continue
                    
                    # Has next lane.
                    has_next_lane = len(self.agents_route[scene_idx][agent_id]) > 1
                    
                    if not has_next_lane:
                        continue
                    
                    # Get the agent's current position.
                    position = self.agents_position[scene_idx][agent_id]
                    
                    # Get the next lane's id.
                    next_lane_id = self.agents_route[scene_idx][agent_id][1]
                    
                    # Check if the agent is in its next lane. 
                    in_next_lane = self.map_api.in_lane_bounds(position, next_lane_id)
                    
                    # The first lane of the route is not the current lane anymore, remove it.
                    if in_next_lane:
                        self.agents_route[scene_idx][agent_id].popleft()

    def update_ego_speed(self, data_batch: Dict[str, torch.Tensor]):
        num_of_scenes = len(data_batch['scene_index'])

        for scene_idx in range(num_of_scenes):
            # Get the ego reference system to world reference system transformation matrix.
            world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
            
            if scene_idx not in self.ego_speed:
                # Get the initial speed of the ego in the scene.
                self.ego_speed[scene_idx] = data_batch['speed'][scene_idx].cpu().numpy()
            else:
                # Get the position of the ego in the scene.
                position = self.ego_position[scene_idx]
                
                # Get the previous position of the ego in the scene.
                previous_position = data_batch["history_positions"][scene_idx][1].cpu().numpy()
                
                # Transform the position to the world reference system.
                previous_position = transform_point(previous_position, world_from_ego)
                
                # Calculate the ego's speed.
                speed = np.linalg.norm(position - previous_position) / self.timestep
                
                # Assign the ego's speed in the scene.
                self.ego_speed[scene_idx] = speed

    def update_agents_speed(self, data_batch: Dict[str, torch.Tensor]):
        num_of_scenes = len(data_batch['scene_index'])
        
        for scene_idx in range(num_of_scenes):
            # Get the ego reference system to world reference system transformation matrix.
            world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
            
            # Initialize the agent route dictionary for each scene.
            if scene_idx not in self.agents_speed:
                self.agents_speed[scene_idx] = {}
            
            # Get the list of agent id's in this scene.
            agents_ids = data_batch["all_other_agents_track_ids"][scene_idx].cpu().numpy()
            
            for agent_idx, agent_id in enumerate(agents_ids):
                # Ensure the agent has a valid id (an id of 0 is invalid).
                if agent_id == 0:
                    continue
                
                if agent_id not in self.agents_speed[scene_idx]:
                    # Set the initial speed of the agent in the scene.
                    self.agents_speed[scene_idx][agent_id] = 0.0
                else:
                    # Get the availability of the agent in the scene's frames.
                    availability = data_batch["all_other_agents_history_availability"][scene_idx][agent_idx].cpu().numpy()
                
                    # Ensure the agent's historical positions are known (are available).
                    if not all(availability):
                        continue
                    
                     # Get the position of the agent in the scene.
                    position = self.agents_position[scene_idx][agent_id]
                    
                    # Get the previous position of the agent in the scene.
                    previous_position = data_batch["all_other_agents_history_positions"][scene_idx][agent_idx][1].cpu().numpy()
                    
                    # Transform the position to the world reference system.
                    previous_position = transform_point(previous_position, world_from_ego)
                    
                    # Calculate the agent's speed.
                    speed = np.linalg.norm(position - previous_position) / self.timestep
                    
                    # Set the agent's speed for this scene.
                    self.agents_speed[scene_idx][agent_id] = speed

    def update_ego_leader(self, data_batch: Dict[str, torch.Tensor]):
        num_of_scenes = len(data_batch['scene_index'])
        
        for scene_idx in range(num_of_scenes):

            self.ego_leader[scene_idx] = None

            for _, (agent_id, agent_route) in enumerate(self.agents_route[scene_idx].items()):
                # Ensure the agent has a route.
                if agent_route == None: continue
                
                 # Ensure the ego and agent share one or more lanes in their route.
                if set(self.ego_route[scene_idx]).isdisjoint(set(agent_route)): continue

                # Get the agent's current position.
                agent_local_position = self.agents_local_position[scene_idx][agent_id]

                # Ensure the agent is in front the ego.
                if agent_local_position[0] < 0: continue
                
                # Determine the agent's distance to the ego.
                agent_distance = np.linalg.norm(agent_local_position)
                
                # Update the ego's leader.
                if self.ego_leader[scene_idx] == None:
                    self.ego_leader[scene_idx] = agent_id
                    self.ego_leader_distance[scene_idx] = agent_distance
                elif self.ego_leader_distance[scene_idx] > agent_distance:
                    self.ego_leader[scene_idx] = agent_id
                    self.ego_leader_distance[scene_idx] = agent_distance

    def update_traffic_lights(self, data_batch: Dict[str, torch.Tensor]):
        num_of_scenes = len(data_batch['scene_index'])
        
        for scene_idx in range(num_of_scenes):
            # Initialize this scene's dict which stores the traffic light status of the lanes on the ego's route.
            self.traffic_lights[scene_idx] = {}
            
            # Decode the list of active traffic light faces id's in this scene.
            traffic_light_faces_ids = data_batch["traffic_light_faces_ids"][scene_idx].cpu().numpy()
            traffic_light_faces_ids = [self.map_api.int_as_id(face_id) for face_id in traffic_light_faces_ids]
            
            # Retrieve the color of these traffic light faces.
            traffic_light_faces_colors = data_batch["traffic_light_faces_colors"][scene_idx].cpu().numpy()
            
            for lane_id in self.ego_route[scene_idx]:
                # Get the traffic control id's for this lane.
                traffic_control_ids = self.map_api.get_lane_traffic_control_ids(lane_id)
                
                for idx, traffic_control_id in enumerate(traffic_control_ids):
                    # If the traffic control id is found in the list of active traffic lights, save this lane's traffic
                    # light color (red, yellow or green).
                    if traffic_control_id in traffic_light_faces_ids:
                        self.traffic_lights[scene_idx][lane_id] = traffic_light_faces_colors[idx]
