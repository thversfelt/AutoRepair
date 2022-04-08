import random
from time import time
import numpy as np
import torch

from typing import Deque, Dict
from l5kit.geometry.transform import transform_point, transform_points, yaw_as_rotation33
from autotest.util.map_api import CustomMapAPI


class Agent:
    
    def __init__(self, map: CustomMapAPI, scene_index: int):
        self.map: CustomMapAPI= map
        self.scene_index: int = scene_index
        
        self.world_from_ego = None
        self.ego_from_world = None
        
        self.position: np.ndarray = None
        self.yaw: float = None
        self.extent: np.ndarray = None
        self.local_velocity: np.ndarray = None
        self.velocity: np.ndarray = None
        self.speed: float = None
        
        self.route: Deque[str] = None
        
    def update(self, data_batch: Dict[str, torch.Tensor]):
        self.update_transformation_matrices(data_batch)

    def update_transformation_matrices(self, data_batch: Dict[str, torch.Tensor]):
        self.world_from_ego = data_batch["world_from_agent"][self.scene_index].cpu().numpy()
        self.ego_from_world = data_batch["agent_from_world"][self.scene_index].cpu().numpy()

class VehicleAgent(Agent):
    
    def __init__(self, map: CustomMapAPI, scene_index: int, id: int):
        super().__init__(map, scene_index)
        
        self.id: int = id
        self.parked: bool = False
        self.local_yaw: np.ndarray = None
        self.local_position: np.ndarray = None

    def update(self, data_batch: Dict[str, torch.Tensor], index: int):
        super().update(data_batch)
        self.index = index
        
        self.update_position(data_batch)
        self.update_yaw(data_batch)
        self.update_extent(data_batch)
        self.update_velocity(data_batch)
        self.update_route(data_batch)
        
    def update_position(self, data_batch: Dict[str, torch.Tensor]):
        # Get the agent's local position.
        self.local_position = data_batch["all_other_agents_history_positions"][self.scene_index][self.index][0].cpu().numpy()

        # Transform the position to the world reference system.
        self.position = transform_point(self.local_position, self.world_from_ego)
    
    def update_yaw(self, data_batch: Dict[str, torch.Tensor]):
        self.local_yaw = data_batch["all_other_agents_history_yaws"][self.scene_index][self.index][0].cpu().numpy()
        ego_yaw = data_batch["yaw"][self.scene_index].cpu().numpy()
        self.yaw = ego_yaw + self.local_yaw
        
    def update_extent(self, data_batch: Dict[str, torch.Tensor]):
        self.extent = data_batch["all_other_agents_history_extents"][self.scene_index][self.index][0].cpu().numpy()
    
    def update_velocity(self, data_batch: Dict[str, torch.Tensor]):
        # Get the availability of the agent in the scene's frames.
        availability = data_batch["all_other_agents_history_availability"][self.scene_index][self.index].cpu().numpy()

        # Ensure the agent's historical positions are known (are available).
        if not all(availability):
            self.speed = 0.0
            self.local_velocity = np.array([self.speed, 0.0])
            self.velocity = np.dot(self.world_from_ego[:2, :2], self.local_velocity)
            return

        # Get the simulator's time step size.
        timestep = 0.1

        # Get the previous position of the agent in the scene.
        previous_local_position = data_batch["all_other_agents_history_positions"][self.scene_index][self.index][1].cpu().numpy()
        
        # Calculate the ego's local velocity.
        self.local_velocity = (self.local_position - previous_local_position) / timestep

        # Calculate the agent's velocity.
        self.velocity = np.dot(self.world_from_ego[:2, :2], self.local_velocity)
        
        # Calculate the agent's speed.
        self.speed = np.linalg.norm(self.velocity)

    def update_route(self, data_batch: Dict[str, torch.Tensor]):
        if self.route is None and self.parked == False:
            self.determine_route(data_batch)
        else:
            self.adjust_route()
        
    def determine_route(self, data_batch: Dict[str, torch.Tensor]):
        # Get the availability of the agent in the scene's frames.
        availability = data_batch["all_other_agents_future_availability"][self.scene_index][self.index].cpu().numpy()
        
        # Get the trajectory of the agent in the scene.
        trajectory = data_batch["all_other_agents_future_positions"][self.scene_index][self.index].cpu().numpy()
        
        # Filter the trajectory based on that agent's availability for each frame.
        trajectory = trajectory[availability]
        
        # Transform the trajectory to the world reference system.
        trajectory = transform_points(trajectory, self.world_from_ego)
        
        # Get the route that matches the trajectory.
        self.route = self.map.get_route(trajectory)
        
        # If no route could be found, the vehicle doesn't appear to move.
        if self.route is None:
            self.parked = True
        
    def adjust_route(self):
        # Ensure the agent has a route.
        if self.route is None:
            return
        
        # Has next lane.
        has_next_lane = len(self.route) > 1
        
        # Ensure there are at least two lanes (the current lane, and the next lane).
        if not has_next_lane:
            return
        
        # Get the next lane's id.
        next_lane_id = self.route[1]
        
        # Check if the agent is in its next lane. 
        in_next_lane = self.map.in_lane(self.position, next_lane_id)
        
        # The first lane of the route is not the current lane anymore, remove it.
        if in_next_lane:
            self.route.popleft()
            
class EgoAgent(Agent):

    def __init__(self, map: CustomMapAPI, scene_index: int):
        super().__init__(map, scene_index)

        self.length: float = None
        self.width: float = None
        
        self.leader: VehicleAgent = None
        self.time_to_collision: float = None
        self.traffic_light = None

    def update(self, data_batch: Dict[str, torch.Tensor], agents: Dict[int, VehicleAgent]):
        super().update(data_batch)
        
        self.update_position(data_batch)
        self.update_yaw(data_batch)
        self.update_extent(data_batch)
        self.update_velocity(data_batch)
        self.update_route(data_batch)
        self.update_leader(agents)
        self.update_time_to_collision(agents)
        self.update_traffic_light(data_batch)

    def update_position(self, data_batch: Dict[str, torch.Tensor]):
        self.position = data_batch["centroid"][self.scene_index].cpu().numpy()
    
    def update_yaw(self, data_batch: Dict[str, torch.Tensor]):
        self.yaw = data_batch["yaw"][self.scene_index].cpu().numpy()
    
    def update_extent(self, data_batch: Dict[str, torch.Tensor]):
        self.extent = data_batch["extent"][self.scene_index][:2].cpu().numpy()
        self.length = self.extent[0]
        self.width = self.extent[1]
    
    def update_velocity(self, data_batch: Dict[str, torch.Tensor]):   
        # Get the availability of the ego in the scene's frames.
        availability = data_batch["history_availabilities"][self.scene_index].cpu().numpy()
        
        # Ensure the ego's historical positions are known (are available).
        if not all(availability):
            self.speed = data_batch['speed'][self.scene_index].cpu().numpy()
            self.local_velocity = np.array([self.speed, 0.0])
            self.velocity = np.dot(self.world_from_ego[:2, :2], self.local_velocity)
            return
        
        # Get the simulator's time step size.
        timestep = 0.1
        
        # Get the previous local position of the ego in the scene.
        previous_local_position = data_batch["history_positions"][self.scene_index][1].cpu().numpy()

        # Calculate the ego's local velocity.
        self.local_velocity = -previous_local_position / timestep

        # Calculate the ego's velocity.
        self.velocity = np.dot(self.world_from_ego[:2, :2], self.local_velocity)

        # Calculate the ego's speed.
        self.speed = np.linalg.norm(self.velocity)
        
    def update_route(self, data_batch: Dict[str, torch.Tensor]):
        if self.route is None:
            self.determine_route(data_batch)
        else:
            self.adjust_route()
    
    def determine_route(self, data_batch: Dict[str, torch.Tensor]):
        # Get the availability of the ego in the scene's frames.
        availability = data_batch["target_availabilities"][self.scene_index].cpu().numpy()
        
        # Get the trajectory of the ego in the scene.
        trajectory = data_batch["target_positions"][self.scene_index].cpu().numpy()
        
        # Filter the trajectory based on the ego's availability for each frame.
        trajectory = trajectory[availability]
        
        # Transform the trajectory to the world reference system.
        trajectory = transform_points(trajectory, self.world_from_ego)
        
        # Get the route that matches the trajectory.
        self.route = self.map.get_route(trajectory)

        # Ensure there are at least two lanes (the current lane, and the next lane), otherwise extend the route.
        if len(self.route) <= 1:
            self.extend_route()
        
    def adjust_route(self):
        # Get the next lane's id.
        next_lane_id = self.route[1]
        
        # Check if the ego is in its next lane. 
        in_next_lane = self.map.in_lane(self.position, next_lane_id)
        
        # The first lane of the route is not the current lane anymore, remove it from the route.
        if in_next_lane:
            self.route.popleft()

        # Ensure there are at least two lanes (the current lane, and the next lane), otherwise extend the route.
        if len(self.route) <= 1:
            self.extend_route()
            
    def extend_route(self):
        current_lane_id = self.route[0]
        ahead_lanes_ids = self.map.get_ahead_lanes_ids(current_lane_id)
        change_lanes_ids = self.map.get_change_lanes_ids(current_lane_id)
        
        if len(ahead_lanes_ids) > 0:
            next_lane_id = random.choice(ahead_lanes_ids)
        elif len(change_lanes_ids) > 0:
            next_lane_id = random.choice(change_lanes_ids)
        else:
            next_lane_id = current_lane_id
        
        self.route.append(next_lane_id)

    def update_leader(self, agents: Dict[int, VehicleAgent]):
        self.leader = None
        
        for _, agent in agents.items():
            # Ensure the agent has a route.
            if agent.route is None: 
                continue
            
            # Ensure the ego and agent share one or more lanes in their route.
            if set(self.route).isdisjoint(set(agent.route)): 
                continue

            # Ensure the agent is in front the ego.
            if agent.local_position[0] < 0: 
                continue
            
            # Set as the ego's leader if there is none.
            if self.leader is None:
                self.leader = agent
                continue
            
            # Determine the agent's distance to the ego.
            agent_distance_to_ego = np.linalg.norm(agent.local_position)
            
            # Determine the leader's distance to the ego.
            leader_distance_to_ego = np.linalg.norm(self.leader.local_position)
            
            if agent_distance_to_ego < leader_distance_to_ego:
                self.leader = agent
    
    def update_time_to_collision(self, agents: Dict[int, VehicleAgent]):
        self.time_to_collision = 99999
        
        for agent in agents.values():
            
            # Ensure the agent is in front of the ego, otherwise ignore this agent.
            if agent.local_position[0] < 0:
                continue
            
            # Calculate the time to collision to this agent from the ego in each direction by linearly extrapolating the 
            # ego's and agent's position using their velocity.
            time_to_collision = (agent.position - self.position) / (self.velocity - agent.velocity)
            
            # Ensure the time-to-collision in each direction is positive, otherwise ignore this agent.
            if time_to_collision[0] < 0 or time_to_collision[1] < 0:
                continue
            
            # Calculate the time difference between the time-to-collision of each direction.
            time_difference = abs(time_to_collision[0] - time_to_collision[1])
            
            # Calculate the minimum time to collision.
            min_time_to_collision = min(time_to_collision)
            
            # If the time difference between the directions is smaller than 0.5 seconds, the ego and agent are projected
            # to collide in the future, with a margin of 0.5s between the time of collision in the x-direction and 
            # y-direction.
            if time_difference < 1.0 and min_time_to_collision < self.time_to_collision:
                self.collider_id = agent.id
                self.time_to_collision = min_time_to_collision
    
    def update_traffic_light(self, data_batch: Dict[str, torch.Tensor]):
        self.traffic_light = None
        
        # Decode the list of active traffic light faces id's in this scene.
        traffic_light_faces_ids = data_batch["traffic_light_faces_ids"][self.scene_index].cpu().numpy()
        traffic_light_faces_ids = [self.map.int_as_id(face_id) for face_id in traffic_light_faces_ids]
        
        traffic_light_to_color: Dict[str, str] = {}
        for face_id in traffic_light_faces_ids:
            try:
                traffic_light_to_color[face_id] = self.map.get_color_for_face(face_id).lower()
            except KeyError:
                continue
        
        current_lane_id = self.route[0]
        self.traffic_light = self.map.get_tl_feature_for_lane(current_lane_id, traffic_light_to_color)
