import random
from l5kit.geometry.transform import transform_point, transform_points
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
from benchmark.custom_map_api import CustomMapAPI
from benchmark.scene import Scene


class EgoModelPerception():
    def __init__(self, map: CustomMapAPI):
        self.map = map
        self.scenes: Dict[Scene] = {}

    def process(self, data_batch: Dict[str, torch.Tensor]):
        scene_ids = data_batch['scene_index'].cpu().numpy()
        for scene_index, scene_id in enumerate(scene_ids):
            if scene_index not in self.scenes:
                self.scenes[scene_index] = Scene(scene_index, scene_id, self.map)
            scene = self.scenes[scene_index]
            scene.update(data_batch)

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
