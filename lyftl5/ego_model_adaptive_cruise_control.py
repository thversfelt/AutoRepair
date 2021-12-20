from typing import Dict
from l5kit.geometry.transform import transform_point
import numpy as np
import torch
from torch import nn
from lyftl5.custom_map_api import CustomMapAPI


class EgoModelAdaptiveCruiseControl(nn.Module):
    def __init__(self, map_api: CustomMapAPI):
        super().__init__()
        self.map_api = map_api
        self.timestep: float = 0.1  # [s]

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        num_of_scenes = len(data_batch['scene_index'])
        num_of_agents = len(data_batch['all_other_agents_types'][0])
        
        acc = torch.zeros([num_of_scenes], dtype=torch.float64)
        
        for scene_idx in range(num_of_scenes):
            ego_position = data_batch["centroid"][scene_idx].cpu().numpy()
            ego_lane_id = self.map_api.get_closest_lanes_ids(ego_position)[0]
            ego_lane_progress = self.map_api.get_lane_progress(ego_position, ego_lane_id)
            
            world_from_ego = data_batch["world_from_agent"][scene_idx].cpu().numpy()
            ego_speed = data_batch["speed"][scene_idx].cpu().numpy()
            
            leading_agent_idx = -1
            leading_agent_lane_progress = -1
            leading_agent_speed = -1
            
            for agent_idx in range(num_of_agents):
                # Ensure the agent is available (exists) during this timestep and the previous timestep.
                agent_availability = data_batch['all_other_agents_history_availability'][scene_idx][agent_idx][0]
                agent_previous_availability = data_batch['all_other_agents_history_availability'][scene_idx][agent_idx][1]
                if not agent_availability or not agent_previous_availability: continue
                
                agent_position = data_batch['all_other_agents_history_positions'][scene_idx][agent_idx][0].cpu().numpy()
                agent_position = transform_point(agent_position, world_from_ego)
                agent_line_id = self.map_api.get_closest_lanes_ids(agent_position)[0]
                
                # Ensure the agent is in the lane that the ego is also in.
                if not agent_line_id == ego_lane_id: continue
                
                agent_previous_position = data_batch['all_other_agents_history_positions'][scene_idx][agent_idx][1].cpu().numpy()
                agent_previous_position = transform_point(agent_previous_position, world_from_ego)
                agent_velocity = (agent_position - agent_previous_position) / self.timestep
                agent_speed = np.linalg.norm(agent_velocity)
                
                # TODO: ignore parked cars. Requires some sort of detection of parked cars.
                
                # Get the progress of the agent in the lane, where 0 is at the beginning and 1 is at the end.
                agent_lane_progress = self.map_api.get_lane_progress(agent_position, agent_line_id)
                
                # Ensure the agent is ahead of the ego in the lane.
                if not (agent_lane_progress > ego_lane_progress): continue

                # Update the leading agent if this leading agent is closer to the ego.
                if agent_lane_progress > leading_agent_lane_progress or leading_agent_lane_progress == -1:
                    leading_agent_idx = agent_idx
                    leading_agent_lane_progress = agent_lane_progress
                    leading_agent_speed = agent_speed
            
            if leading_agent_idx == -1:
                acc[scene_idx] = 3.0
            elif leading_agent_lane_progress > ego_lane_progress:
                acc[scene_idx] = 3.0
            else:
                acc[scene_idx] = -3.0

        eval_dict = {"acc": acc}
        return eval_dict
            