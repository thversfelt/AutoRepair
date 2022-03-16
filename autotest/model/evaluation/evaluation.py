from typing import List

import numpy as np
from autotest.model.context.scene import Scene
from autotest.model.evaluation.metrics import Metric


class Evaluation:
    
    def __init__(self, metrics: List[Metric]) -> None:
        self.metrics: List[Metric] = metrics
        self.results: dict = {}
        
    def process(self, scene: Scene, predicted_position: np.ndarray, predicted_yaw: float):
        for metric in self.metrics:
            if scene.id not in self.results:
                self.results[scene.id] = {}
            
            if metric.name not in self.results[scene.id]:
                self.results[scene.id][metric.name] = []
            
            score = metric.process(scene, predicted_position, predicted_yaw)
            self.results[scene.id][metric.name].append(score)
