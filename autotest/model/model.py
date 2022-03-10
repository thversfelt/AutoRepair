import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
from autotest.model.evaluation.evaluation import Evaluation
from autotest.model.evaluation.metrics import Metric
from autotest.model.modules.control import Control
from autotest.model.modules.navigation import Navigation
from autotest.model.modules.perception import Perception
from autotest.model.modules.planning import Planning
from autotest.util.map_api import CustomMapAPI


class Model(nn.Module):
    
    def __init__(self, map: CustomMapAPI, metrics: List[Metric]):
        super().__init__()
        self.map = map
        self.metrics = metrics
        self.reset()

    def reset(self):
        self.perception = Perception(self.map)
        self.evaluation = Evaluation(self.metrics)
        
    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.perception.process(data_batch)

        positions = []
        yaws = []
    
        for _, scene in self.perception.scenes.items():
            self.evaluation.evaluate(scene)
            position, yaw = Planning().process(scene)
            positions.append(position)
            yaws.append(yaw)

        positions = torch.tensor(positions)
        yaws = torch.tensor(yaws)
        
        num_of_scenes = len(self.perception.scenes)
        return {
            "positions": torch.reshape(positions, [num_of_scenes, 1, 2]),
            "yaws": torch.reshape(yaws, [num_of_scenes, 1, 1])
        }
