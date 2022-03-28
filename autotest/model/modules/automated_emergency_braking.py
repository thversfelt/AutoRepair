import numpy as np

from autotest.model.context.scene import Scene


class AutomatedEmergencyBraking():
    
    ACCELERATION = 1.0 # [m/s^2]
    DECELLERATION = -5.0 # [m/s^2]

    def process(self, scene: Scene) -> float:
        if scene.ego.local_velocity[0] > 0:
            return self.DECELLERATION
        else:
            return self.ACCELERATION
