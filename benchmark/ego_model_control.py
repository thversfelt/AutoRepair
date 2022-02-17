import math
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from benchmark.ego_model_perception import EgoModelPerception

class EgoModelControl(nn.Module):
    def __init__(self, perception: EgoModelPerception):
        super().__init__()
        self.perception = perception
        self.min_acc: float = -3.0  # Min acceleration [m/s^2]
        self.max_acc: float = 3.0  # Max acceleration [m/s^2]
        self.min_steer: float = -math.radians(45)  # Min yaw rate [rad/s]
        self.max_steer: float = math.radians(45)  # Max yaw rate [rad/s]

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Implements The Kinematic Bicycle Model: a Consistent Model for Planning Feasible Trajectories for Autonomous
                Vehicles? (2017)"""
        num_of_scenes = len(data_batch['scene_index'])

        # The x,y positions and yaws of the ego agent's reference system in each scene.
        positions = np.zeros([num_of_scenes, 2])
        yaws = np.zeros([num_of_scenes, 1])

        for i in range(num_of_scenes):
            speed = self.perception.ego_speed[i]
            timestep = self.perception.timestep
            
            length = data_batch['extent'][i][0].cpu().numpy()
            steer = data_batch['steer'][i].cpu().numpy()
            acc = data_batch['acc'][i].cpu().numpy()

            steer = np.clip(steer, self.min_steer, self.max_steer)
            acc = np.clip(acc, self.min_acc, self.max_acc)
            
            speed += acc * timestep
            beta = np.arctan(0.5 * np.tan(steer))
            velocity = speed * np.array([
                np.cos(beta),
                np.sin(beta)
            ])

            position = velocity * timestep
            yaw = speed * np.sin(beta) / (0.5 * length) * timestep

            positions[i] = position
            yaws[i] = yaw

        positions = torch.from_numpy(positions)
        yaws = torch.from_numpy(yaws)
        
        eval_dict = {
            "positions": torch.reshape(positions, [num_of_scenes, 1, 2]),
            "yaws": torch.reshape(yaws, [num_of_scenes, 1, 1])
        }

        return eval_dict
