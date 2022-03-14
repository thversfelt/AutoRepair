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

        predicted_positions = []
        predicted_yaws = []
    
        for _, scene in self.perception.scenes.items():
            predicted_position, predicted_yaw = Planning().process(scene)
            self.evaluation.evaluate(scene, predicted_position, predicted_yaw)
            
            predicted_positions.append(predicted_position)
            predicted_yaws.append(predicted_yaw)
            
        predicted_positions = torch.tensor(np.array(predicted_positions))
        predicted_yaws = torch.tensor(np.array(predicted_yaws))
        
        num_of_scenes = len(self.perception.scenes)
        return {
            "positions": torch.reshape(predicted_positions, [num_of_scenes, 1, 2]),
            "yaws": torch.reshape(predicted_yaws, [num_of_scenes, 1, 1])
        }
