import numpy as np

from l5kit.planning.utils import _get_bounding_box, _get_sides
from l5kit.geometry.transform import transform_point
from autotest.model.context.scene import Scene
from autotest.model.modules.traffic_lights import TrafficLights


class Metric:
    name = ""
    
    def evaluate(self, scene: Scene, predicted_position: np.ndarray, predicted_yaw: float) -> float:
        self.ego_predicted_position = transform_point(predicted_position, scene.ego.world_from_ego)
        self.ego_predicted_yaw = scene.ego.yaw + predicted_yaw


class CollisionMetric(Metric):
    name = "collision"
    
    def evaluate(self, scene: Scene, predicted_position: np.ndarray, predicted_yaw: float) -> float:
        super().evaluate(scene, predicted_position, predicted_yaw)
        
        min_score = 0.0
        
        ego_bounding_box = _get_bounding_box(self.ego_predicted_position, self.ego_predicted_yaw, scene.ego.extent)
        ego_area = scene.ego.length * scene.ego.width
        
        for _, agent in scene.agents.items():
            
            agent_bounding_box = _get_bounding_box(agent.position, agent.yaw, agent.extent)
            intersection = agent_bounding_box.intersection(ego_bounding_box)
            score = -1.0 * intersection.area / ego_area
            
            if score < min_score:
                min_score = score

        return np.clip(min_score, -1.0, 0.0)


class SafeDistanceMetric(Metric):
    name = "safe_distance"
    
    def evaluate(self, scene: Scene, predicted_position: np.ndarray, predicted_yaw: float) -> float:
        super().evaluate(scene, predicted_position, predicted_yaw)
        
        score = 0.0
        
        return np.clip(score, -1.0, 0.0)
    

class TrafficLightsMetric(Metric):
    name = "traffic_lights"
    
    def evaluate(self, scene: Scene, predicted_position: np.ndarray, predicted_yaw: float) -> float:
        super().evaluate(scene, predicted_position, predicted_yaw)
        
        score = 0.0
        
        current_lane_id = scene.ego.route[0]
        next_lane_id = scene.ego.route[1]
        
        in_next_lane = scene.map.in_lane(self.ego_predicted_position, next_lane_id)

        if in_next_lane and scene.ego.traffic_light == TrafficLights.RED:
            score = -1.0 * scene.ego.speed / scene.map.get_lane_speed_limit(current_lane_id)
            
        return np.clip(score, -1.0, 0.0)
    