import numpy as np

from autotest.model.context.scene import Scene


class AdaptiveCruiseControl():
    
    JUNCTION_SPEED = 5.0 # [m/s]
    FOLLOW_DISTANCE = 5.0 # [m]
    ACCELERATION = 1.0 # [m/s^2]
    DECELLERATION = -1.0 # [m/s^2]

    def process(self, scene: Scene) -> float:
        # Get the ego's current lane.
        current_lane_id = scene.ego.route[0]
        
        # Junctions have no recorded speed limit in level 5's dataset, so check if the ego is currently in a junction.
        if scene.map.is_lane_in_junction(current_lane_id):
            # The ego is on a junction, set the speed limit as the junction speed parameter.
            speed_limit = self.JUNCTION_SPEED
        else:
            # The ego is not on a junction, get the speed limit of the lane it is on.
            speed_limit = scene.map.get_lane_speed_limit(current_lane_id)
        
        # If the ego has no leader, just follow the speed limit.
        if scene.ego.leader is None and scene.ego.speed < speed_limit:
            return self.ACCELERATION
        elif scene.ego.speed >= speed_limit:
            return self.DECELLERATION

        # Determine the leader's distance to the ego.
        distance = np.linalg.norm(scene.ego.leader.local_position)
        
        # Determine the leader's relative speed.
        relative_speed = scene.ego.leader.speed - scene.ego.speed

        # The ego has a leader, match its speed as long as it is below the speed limit. 
        if distance >= self.FOLLOW_DISTANCE and relative_speed > 0 and scene.ego.speed < speed_limit:
            return self.ACCELERATION
        else:
            return self.DECELLERATION
