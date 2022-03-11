import math
import numpy as np

from autotest.model.context.scene import Scene


class Control:
    MIN_ACCELERATION: float = -8.0  # Min acceleration (max deceleration) [m/s^2]
    MAX_ACCELERATION: float = 3.0  # Max acceleration [m/s^2]
    MIN_STEER: float = -math.radians(45.0)  # Min yaw rate [rad/s]
    MAX_STEER: float = math.radians(45.0)  # Max yaw rate [rad/s]

    def process(self, scene: Scene, steer: float, acceleration: float) -> tuple:
        """Implements The Kinematic Bicycle Model: a Consistent Model for Planning Feasible Trajectories for Autonomous
                Vehicles? (2017)"""
        steer = np.clip(steer, self.MIN_STEER, self.MAX_STEER)
        acceleration = np.clip(acceleration, self.MIN_ACCELERATION, self.MAX_ACCELERATION)
        
        timestep = 0.1
        speed = scene.ego.speed + acceleration * timestep
        beta = np.arctan(0.5 * np.tan(steer))
        velocity = speed * np.array([
            np.cos(beta),
            np.sin(beta)
        ])

        position = velocity * timestep # In the agent's local space.
        yaw = speed * np.sin(beta) / (0.5 * scene.ego.length) * timestep # The difference in absolute yaw.

        return position, yaw
