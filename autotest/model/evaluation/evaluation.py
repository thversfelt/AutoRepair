from typing import List
from autotest.model.context.scene import Scene
from autotest.model.evaluation.metrics import Metric


class Evaluation:
    
    def __init__(self, metrics: List[Metric]) -> None:
        self.metrics = metrics
        self.results = {}
        
    def evaluate(self, scene: Scene):
        for metric in self.metrics:
            if scene.id not in self.results:
                self.results[scene.id] = {}
            
            if metric.name not in self.results[scene.id]:
                self.results[scene.id][metric.name] = []
            
            score = metric.evaluate(scene)
            self.results[scene.id][metric.name].append(score)
    