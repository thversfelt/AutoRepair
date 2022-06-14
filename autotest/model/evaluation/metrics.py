import numpy as np

from l5kit.planning.utils import _get_bounding_box
from l5kit.geometry.transform import transform_point
from autotest.model.context.scene import Scene
from autotest.model.modules.traffic_lights import TrafficLights


class Metric:
    name = ""
    
    def process(self, scene: Scene, predicted_position: np.ndarray, predicted_yaw: float) -> float:
        self.ego_predicted_position = transform_point(predicted_position, scene.ego.world_from_ego)
        self.ego_predicted_yaw = scene.ego.yaw + predicted_yaw


class CollisionMetric(Metric):
    name = "collision"
    
    def process(self, scene: Scene, predicted_position: np.ndarray, predicted_yaw: float) -> float:
        super().process(scene, predicted_position, predicted_yaw)
        
        min_score = 0.0
        
        ego_bounding_box = _get_bounding_box(self.ego_predicted_position, self.ego_predicted_yaw, scene.ego.extent)
        
        for agent in scene.agents.values():
            
            # Only consider frontal collision (agents ahead of the ego).
            if agent.local_velocity[0] > 0:
                continue
            
            agent_bounding_box = _get_bounding_box(agent.position, agent.yaw, agent.extent)
            
            # Check if the ego and agent are already colliding.
            #if intersection.area > 0:
            #    continue
            
            intersection = agent_bounding_box.intersection(ego_bounding_box)

            # Check if the ego and agent are colliding in the ego's predicted position.
            if intersection.area == 0:
                continue
            
            relative_speed = abs(scene.ego.speed - agent.speed)
            score = -1.0 * relative_speed / (relative_speed + 1)
            
            if score < min_score:
                min_score = score

        return np.clip(min_score, -1.0, 0.0)


class SafeDistanceMetric(Metric):
    name = "safe_distance"
    
    def process(self, scene: Scene, predicted_position: np.ndarray, predicted_yaw: float) -> float:
        super().process(scene, predicted_position, predicted_yaw)
        
        score = 0.0
        
        if scene.ego.leader is not None:
            # Determine the braking distance https://en.wikipedia.org/wiki/Braking_distance
            braking_distance = scene.ego.speed * 1.5 + scene.ego.speed**2 / (2 * 0.7 * 9.81)
            
            # Determine the distance between the ego and its leader.
            leader_distance = np.linalg.norm(scene.ego.leader.local_position)
            
            # The score dips below zero as soon as the leader's distance is lower than the braking distance.
            score = -1.0 * (1.0 - leader_distance / braking_distance)
        
        return np.clip(score, -1.0, 0.0)
    

class TrafficLightsMetric(Metric):
    name = "traffic_lights"
    
    def process(self, scene: Scene, predicted_position: np.ndarray, predicted_yaw: float) -> float:
        super().process(scene, predicted_position, predicted_yaw)
        
        score = 0.0
        
        current_lane_id = scene.ego.route[0]
        next_lane_id = scene.ego.route[1]
        
        in_next_lane = scene.map.in_lane(self.ego_predicted_position, next_lane_id)

        if in_next_lane and scene.ego.traffic_light == TrafficLights.RED:
            score = -1.0 * scene.ego.speed / scene.map.get_lane_speed_limit(current_lane_id)
            
        return np.clip(score, -1.0, 0.0)
    