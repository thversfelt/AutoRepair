import numpy as np

from typing import List
from autotest.model.context.scene import Scene
from autotest.model.evaluation.metrics import Metric


class Evaluation:
    
    def __init__(self, metrics: List[Metric]) -> None:
        self.metrics: List[Metric] = metrics
        self.metrics_scores: dict = {}
        
    def process(self, scene: Scene, predicted_position: np.ndarray, predicted_yaw: float):
        for metric in self.metrics:
            if scene.id not in self.metrics_scores:
                self.metrics_scores[scene.id] = {}
            
            if metric.name not in self.metrics_scores[scene.id]:
                self.metrics_scores[scene.id][metric.name] = []
            
            score = metric.process(scene, predicted_position, predicted_yaw)
            self.metrics_scores[scene.id][metric.name].append(score)
