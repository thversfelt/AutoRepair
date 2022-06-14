from autotest.model.context.scene import Scene
from autotest.model.modules.adaptive_cruise_control import AdaptiveCruiseControl
from autotest.model.modules.automated_emergency_braking import AutomatedEmergencyBraking
from autotest.model.modules.control import Control
from autotest.model.modules.navigation import Navigation
from autotest.model.modules.traffic_lights import TrafficLights


class RuleSet:

    def process(self, scene: Scene) -> tuple:
        steer = Navigation().process(scene)
        
        # if scene.ego.time_to_collision < 1.6:
        #     acceleration = AutomatedEmergencyBraking().process(scene)
        # else:
        #     if scene.ego.traffic_light == TrafficLights.RED:
        #         acceleration = TrafficLights().process(scene)
        #     else:
        #         acceleration = AdaptiveCruiseControl().process(scene)
        
        acceleration = 1.0
        
        return Control().process(scene, steer, acceleration)
