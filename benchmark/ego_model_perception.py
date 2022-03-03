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
