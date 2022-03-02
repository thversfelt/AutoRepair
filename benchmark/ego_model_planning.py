from benchmark.ego_model_adaptive_cruise_control import EgoModelAdaptiveCruiseControl
from benchmark.ego_model_navigation import EgoModelNavigation
from benchmark.scene import Scene


class EgoModelPlanning():
    def __init__(self):
        self.adaptive_cruise_control = EgoModelAdaptiveCruiseControl()

    def process(self, scene: Scene) -> float:
        acc = self.adaptive_cruise_control.process(scene)
        return acc