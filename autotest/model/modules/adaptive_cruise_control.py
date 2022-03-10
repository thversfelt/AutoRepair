import numpy as np

from autotest.model.context.scene import Scene


class AdaptiveCruiseControl():
    
    FOLLOW_DISTANCE = 10.0 # [m]
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
