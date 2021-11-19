import math
from typing import Dict
import torch
import torch.nn as nn


class EgoModelControl(nn.Module):
    def __init__(self, timestep: torch.Tensor = 0.1):
        super().__init__()
        self.min_acc: float = -3  # Min acceleration [m/s^2]
        self.max_acc: float = 3  # Max acceleration [m/s^2]
        self.min_steer: float = -math.radians(45)  # Min yaw rate [rad/s]
        self.max_steer: float = math.radians(45)  # Max yaw rate [rad/s]
        self.timestep = timestep  # [s]
        self.speed = None

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Implements The Kinematic Bicycle Model: a Consistent Model for Planning Feasible Trajectories for Autonomous
                Vehicles? (2017)"""

        num_of_scenes = len(data_batch['scene_index'])

        # Add the ego agent's initial speed in each scene.
        if self.speed is None:
            self.speed = data_batch['speed']

        # The x,y positions and yaws of the ego agent's reference system in each scene.
        positions = torch.zeros([num_of_scenes, 2], dtype=torch.float64)
        yaws = torch.zeros([num_of_scenes, 1], dtype=torch.float64)

        for i in range(num_of_scenes):
            length = data_batch['extent'][i][0]
            steer = data_batch['steer'][i]
            acc = data_batch['acc'][i]

            steer = torch.clip(steer, self.min_steer, self.max_steer)
            acc = torch.clip(acc, self.min_acc, self.max_acc)

            beta = torch.arctan(0.5 * torch.tan(steer))
            velocity = self.speed[i] * torch.tensor([
                torch.cos(beta),
                torch.sin(beta)
            ])

            position = velocity * self.timestep
            yaw = self.speed[i] * torch.sin(beta) / (0.5 * length) * self.timestep

            positions[i] = position
            yaws[i] = yaw

            self.speed[i] += acc * self.timestep

        eval_dict = {
            "positions": torch.reshape(positions, [2, 1, 2]),
            "yaws": torch.reshape(yaws, [2, 1, 1])
        }

        return eval_dict
