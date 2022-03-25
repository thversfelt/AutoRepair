from autotest.model.context.scene import Scene
from autotest.model.modules.adaptive_cruise_control import AdaptiveCruiseControl
from autotest.model.modules.control import Control
from autotest.model.modules.navigation import Navigation
from autotest.model.modules.traffic_lights import TrafficLights


class RuleSet:

    def process(self, scene: Scene) -> tuple:
        steer = Navigation().process(scene)
        
        if scene.ego.traffic_light != TrafficLights.RED:
            acceleration = TrafficLights().process(scene)
        else:
            acceleration = AdaptiveCruiseControl().process(scene)
        
        return Control().process(scene, steer, acceleration)
