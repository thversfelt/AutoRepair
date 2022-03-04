from benchmark.ego_model_adaptive_cruise_control import EgoModelAdaptiveCruiseControl
from benchmark.ego_model_traffic_lights import EgoModelTrafficLights
from benchmark.scene import Scene


class EgoModelPlanning():
    def __init__(self):
        self.adaptive_cruise_control = EgoModelAdaptiveCruiseControl()
        self.traffic_lights = EgoModelTrafficLights()

    def process(self, scene: Scene) -> float:
        
        if scene.ego.traffic_light == self.traffic_lights.RED:
            return self.traffic_lights.process(scene)
        else:
            return self.adaptive_cruise_control.process(scene)
