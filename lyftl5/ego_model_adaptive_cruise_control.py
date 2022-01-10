from typing import Dict, List
from l5kit.geometry.transform import transform_point
import numpy as np
import torch
from torch import nn
from lyftl5.custom_map_api import CustomMapAPI
from l5kit.data.labels import PERCEPTION_LABELS
from lyftl5.ego_model_perception import EgoModelPerception

PERCEPTION_LABEL_CAR = 3

class EgoModelAdaptiveCruiseControl(nn.Module):
    def __init__(self, perception: EgoModelPerception):
        super().__init__()
        self.perception = perception

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        num_of_scenes = len(data_batch['scene_index'])
      
        acc = torch.zeros([num_of_scenes], dtype=torch.float64)
        for scene_idx in range(num_of_scenes):
            
            ego_route = self.perception.ego_route[scene_idx]
            
            leading_agent_id = None
            leading_agent_distance = None
            leading_agent_speed = None
            
            for agent_id, agent_route in self.perception.agents_route[scene_idx].items():
                
                # Ensure the ego and agent share one or more lanes in their route.
                if set(ego_route).isdisjoint(agent_route):
                    continue
                
                # TODO: Get the agent's direction.
                # TODO: Ensure the agent is pointed in similar direction:.
                # if dot(ego_direction, agent_direction) < 0:
                #     continue
                
                # Get the agent's current position.
                agent_local_position = self.perception.agents_local_position[scene_idx][agent_id]

                # Ensure the agent is ahead of the ego.
                if agent_local_position[0] < 0:
                    continue
                
                # Determine the agent's distance to the ego.
                agent_distance = np.linalg.norm(agent_local_position)
                print(agent_distance)
                
            #if leading_agent_track_id == -1:
            #    acc[scene_idx] = 1.0
            #elif ego_speed < leading_agent_speed:
            #    acc[scene_idx] = 1.5
            #elif ego_local_velocity[0] > 0:
            #    acc[scene_idx] = -1.5

        data_batch["acc"] = acc
        return data_batch
