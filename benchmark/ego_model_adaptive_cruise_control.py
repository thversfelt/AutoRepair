from typing import Dict, List
from l5kit.geometry.transform import transform_point
import numpy as np
import torch
from torch import nn
from benchmark.ego_model_perception import EgoModelPerception


class EgoModelAdaptiveCruiseControl(nn.Module):
    def __init__(self, perception: EgoModelPerception):
        super().__init__()
        self.perception = perception
        
        self.FOLLOW_DISTANCE = 10.0 # [m]
        self.ACCELERATION = 1.0 # [m/s^2]
        self.DECELLERATION = -1.0 # [m/s^2]
        

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        num_of_scenes = len(data_batch['scene_index'])
      
        acc = torch.zeros([num_of_scenes], dtype=torch.float64)
        for scene_idx in range(num_of_scenes):
            
            # Get the id of the leader agent.
            leader_id = self.perception.ego_leader[scene_idx]
            
            # If there is no leader, accelerate.
            if leader_id == None:
                acc[scene_idx] = self.ACCELERATION
                continue
            
            # Get the leader's position, relative to the ego (so, in the ego's reference system).
            leader_local_position = self.perception.agents_local_position[scene_idx][leader_id]
            
            # Get the leader's distance to the ego.
            leader_distance = np.linalg.norm(leader_local_position)
            
            # Get the leader's absolute speed.
            leader_speed = self.perception.agents_speed[scene_idx][leader_id]

            # Get the ego's absolute speed.
            ego_speed = self.perception.ego_speed[scene_idx]

            if leader_distance >= self.FOLLOW_DISTANCE and ego_speed < leader_speed:
                acc[scene_idx] = self.ACCELERATION
            elif ego_speed >= leader_speed:
                acc[scene_idx] = self.DECELLERATION

        data_batch["acc"] = acc
        return data_batch
