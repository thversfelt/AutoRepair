from typing import Dict, List
from l5kit.geometry.transform import transform_point
import numpy as np
import torch
from torch import nn
from benchmark.ego_model_perception import EgoModelPerception
from benchmark.scene import Scene


class EgoModelAdaptiveCruiseControl():
    
    FOLLOW_DISTANCE = 10.0 # [m]
    IDLE = 0.0 # [m/s^2]
    ACCELERATION = 1.0 # [m/s^2]
    DECELLERATION = -1.0 # [m/s^2]

    def process(self, scene: Scene) -> float:
        if scene.ego.leader is None:
            return self.ACCELERATION

        # Determine the leader's distance to the ego.
        leader_distance_to_ego = np.linalg.norm(scene.ego.leader.local_position)
        
        if leader_distance_to_ego >= self.FOLLOW_DISTANCE and scene.ego.speed < scene.ego.leader.speed:
            return self.ACCELERATION
        else:
            return self.DECELLERATION
