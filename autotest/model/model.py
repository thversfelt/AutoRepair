import ast
import numpy as np
import torch
import torch.nn as nn

from typing import Dict, List
from autotest.model.evaluation.evaluation import Evaluation
from autotest.model.evaluation.instrumentation import Instrumentation
from autotest.model.evaluation.metrics import Metric
from autotest.model.modules.perception import Perception
from autotest.util.map_api import CustomMapAPI
from autotest.util.visualization import visualize_scene


class Model(nn.Module):
    
    def __init__(self, map: CustomMapAPI):
        super().__init__()
        self.map = map

    def initialize(self, rule_set: ast.Module, metrics: List[Metric]):
        self.perception = Perception(self.map)
        self.instrumentation = Instrumentation(rule_set)
        self.evaluation = Evaluation(metrics)
        
    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        predicted_positions = []
        predicted_yaws = []
        
        self.perception.process(data_batch)
        
        for _, scene in self.perception.scenes.items():
            predicted_position, predicted_yaw = self.instrumentation.rule_set.process(scene)
            self.evaluation.process(scene, predicted_position, predicted_yaw)
            self.instrumentation.process(scene)
            
            predicted_positions.append(predicted_position)
            predicted_yaws.append(predicted_yaw)
            
        predicted_positions = torch.tensor(np.array(predicted_positions))
        predicted_yaws = torch.tensor(np.array(predicted_yaws))
        
        num_of_scenes = len(self.perception.scenes)
        return {
            "positions": torch.reshape(predicted_positions, [num_of_scenes, 1, 2]),
            "yaws": torch.reshape(predicted_yaws, [num_of_scenes, 1, 1])
        }
