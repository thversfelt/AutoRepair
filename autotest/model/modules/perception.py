import torch

from typing import Dict
from autotest.model.context.scene import Scene
from autotest.util.map_api import CustomMapAPI


class Perception():
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
