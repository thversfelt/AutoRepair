from typing import Dict, List
from l5kit.geometry.transform import transform_point
import numpy as np
import torch
from torch import nn
from lyftl5.custom_map_api import CustomMapAPI
from l5kit.data.labels import PERCEPTION_LABELS

PERCEPTION_LABEL_CAR = 3

class EgoModelAdaptiveCruiseControl(nn.Module):
    def __init__(self, map_api: CustomMapAPI):
        super().__init__()
        self.map_api = map_api
        self.timestep: float = 0.1  # [s]
        self.agents_parked = None
        self.parked_movement_threshold: float = 2.0  # [m]

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        num_of_scenes = len(data_batch['scene_index'])
        num_of_agents = len(data_batch['all_other_agents_types'][0])
        
        if self.agents_parked is None:
            self.agents_parked = torch.zeros([num_of_scenes, num_of_agents], dtype=torch.bool)
            for scene_idx in range(num_of_scenes):
                for agent_idx in range(num_of_agents):
                    agent_availability = data_batch['all_other_agents_future_availability'][scene_idx][agent_idx]
                    agent_available = [i for i, avialable in enumerate(agent_availability) if avialable]
                    if len(agent_available) < 2: continue
                    agent_start_frame_idx = agent_available[0]
                    agent_end_frame_idx = agent_available[-1]
                    agent_start_position = data_batch['all_other_agents_future_positions'][scene_idx][agent_idx][agent_start_frame_idx]
                    agent_end_position = data_batch['all_other_agents_future_positions'][scene_idx][agent_idx][agent_end_frame_idx]
                    agent_movement = torch.linalg.norm(agent_end_position - agent_start_position)
                    if agent_movement > self.parked_movement_threshold: continue
                    self.agents_parked[scene_idx][agent_idx] = True
                        
        acc = torch.zeros([num_of_scenes], dtype=torch.float64)
        
        for scene_idx in range(num_of_scenes):
            # Ensure the ego is available (exists) during this frame and the previous frame.
            ego_availability = data_batch['history_availabilities'][scene_idx][0]
            ego_previous_availability = data_batch['history_availabilities'][scene_idx][1]
            if not ego_availability or not ego_previous_availability: continue
            
            world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
            ego_local_position = data_batch["history_positions"][scene_idx][0].cpu().numpy()
            ego_position = transform_point(ego_local_position, world_from_ego)
            
            ego_previous_local_position = data_batch['history_positions'][scene_idx][1].cpu().numpy()
            ego_previous_position = transform_point(ego_previous_local_position, world_from_ego)
            ego_local_velocity = (ego_local_position - ego_previous_local_position) / self.timestep
            ego_speed = np.linalg.norm(ego_local_velocity)
            
            ego_lane_id = self.map_api.get_closest_lanes_ids(ego_position)[0]
            ego_lane_progress = self.map_api.get_lane_progress(ego_position, ego_lane_id)
            
            leading_agent_track_id = -1
            leading_agent_speed = -1
            leading_agent_gap = -1
            
            for agent_idx in range(num_of_agents):
                # Ensure the agent is available (exists) during this frame and the previous frame.
                agent_availability = data_batch['all_other_agents_history_availability'][scene_idx][agent_idx][0]
                agent_previous_availability = data_batch['all_other_agents_history_availability'][scene_idx][agent_idx][1]
                if not agent_availability or not agent_previous_availability: continue
                
                agent_track_id = data_batch['all_other_agents_track_ids'][scene_idx][agent_idx]
                agent_type = data_batch['all_other_agents_types'][scene_idx][agent_idx]
                
                # Ensure the agent is a car.
                if agent_type != PERCEPTION_LABEL_CAR: continue
                
                agent_local_position = data_batch['all_other_agents_history_positions'][scene_idx][agent_idx][0].cpu().numpy()
                agent_position = transform_point(agent_local_position, world_from_ego)
                agent_lane_id = self.map_api.get_closest_lanes_ids(agent_position)[0]
                
                # Ensure the agent is in the lane that the ego is also in.
                if agent_lane_id != ego_lane_id: continue

                # Ensure the agent is not parked.
                #agent_is_parked = self.agents_parked[scene_idx][agent_idx]
                #if agent_is_parked: continue
                
                # Get the progress of the agent in the lane, where 0 is at the beginning and 1 is at the end.
                # agent_lane_progress = self.map_api.get_lane_progress(agent_position, agent_lane_id)

                # Ensure the agent is ahead of the ego in the lane.
                if agent_local_position[0] < 0: continue

                agent_previous_local_position = data_batch['all_other_agents_history_positions'][scene_idx][agent_idx][1].cpu().numpy()
                agent_previous_position = transform_point(agent_previous_local_position, world_from_ego)
                agent_velocity = (agent_position - agent_previous_position) / self.timestep
                agent_speed = np.linalg.norm(agent_velocity)
                agent_gap = np.linalg.norm(ego_position - agent_position)

                # Ensure this agent is closer to the ego than the current leading agent (if it exists).
                if leading_agent_track_id == -1 or agent_gap < leading_agent_gap: 
                    # Update the leading agent.
                    leading_agent_track_id = agent_track_id
                    leading_agent_speed = agent_speed
                    leading_agent_gap = agent_gap
            
            if leading_agent_track_id == -1:
                acc[scene_idx] = 1.0
            elif leading_agent_gap > 50 and ego_speed < self.map_api.get_lane_speed_limit(ego_lane_id):
                acc[scene_idx] = 1.5
            elif ego_local_velocity[0] > 0:
                acc[scene_idx] = -1.5

        data_batch["acc"] = acc
        return data_batch
