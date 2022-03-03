from typing import Dict, List
from l5kit.geometry.transform import transform_point
import numpy as np
import torch
from torch import nn
from benchmark.ego_model_perception import EgoModelPerception
from benchmark.scene import Scene


class EgoModelTrafficLights():

    UNKNOWN = 0, 
    GREEN = 1
    YELLOW = 2
    RED = 3
    NONE = 4

    ACCELERATION = 1.0 # [m/s^2]
    DECELLERATION = -3.0 # [m/s^2]

    def process(self, scene: Scene) -> float:
        if scene.ego.traffic_light == self.RED and scene.ego.speed > 0.0:
            return self.DECELLERATION
        else:
            return self.ACCELERATION
