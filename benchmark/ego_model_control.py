import math
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from benchmark.agent import EgoAgent
from benchmark.ego_model_perception import EgoModelPerception
from benchmark.scene import Scene

class EgoModelControl:
    def __init__(self):
        self.min_acc: float = -3.0  # Min acceleration [m/s^2]
        self.max_acc: float = 3.0  # Max acceleration [m/s^2]
        self.min_steer: float = -math.radians(45)  # Min yaw rate [rad/s]
        self.max_steer: float = math.radians(45)  # Max yaw rate [rad/s]

    def process(self, ego: EgoAgent, steer: np.float, acc: np.float):
        """Implements The Kinematic Bicycle Model: a Consistent Model for Planning Feasible Trajectories for Autonomous
                Vehicles? (2017)"""
        steer = np.clip(steer, self.min_steer, self.max_steer)
        acc = np.clip(acc, self.min_acc, self.max_acc)
        
        timestep = 0.1
        ego.speed += acc * timestep
        beta = np.arctan(0.5 * np.tan(steer))
        velocity = ego.speed * np.array([
            np.cos(beta),
            np.sin(beta)
        ])

        position = velocity * timestep
        yaw = ego.speed * np.sin(beta) / (0.5 * ego.length) * timestep

        return position, yaw
