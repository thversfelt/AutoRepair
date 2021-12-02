import math
from typing import Dict
import torch
import torch.nn as nn
from lyftl5.custom_map_api import CustomMapAPI


class EgoModelNavigation(nn.Module):
    def __init__(self, map_api: CustomMapAPI):
        super().__init__()
        self.map_api = map_api

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        num_of_scenes = len(data_batch['scene_index'])
        for scene_idx in range(num_of_scenes):
            ego_position = data_batch["centroid"][scene_idx]
            #  current_lane = self.map_api.get_closest_lane(ego_position.cpu().numpy())
        return {}
