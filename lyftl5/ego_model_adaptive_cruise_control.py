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
            
            # If there is no leader agent, just use "normal" acceleration.
            if self.perception.ego_leader[scene_idx] == None:
                acc[scene_idx] = 1.0
                continue
            
            # Get the agent id of the leader.
            agent_id = self.perception.ego_leader[scene_idx]
            
            # 
            if self.perception.ego_speed[scene_idx] < self.perception.agents_speed[scene_idx][agent_id]:
                acc[scene_idx] = 1.5
            else:
                acc[scene_idx] = -1.5

        data_batch["acc"] = acc
        return data_batch
