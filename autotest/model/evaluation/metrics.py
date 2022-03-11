import numpy as np

from l5kit.planning.utils import _get_bounding_box, _get_sides
from l5kit.geometry.transform import transform_point
from autotest.model.context.scene import Scene


class Metric:
    name = ""
    
    def evaluate(self, scene: Scene, predicted_position: np.ndarray, predicted_yaw: float) -> float:
        self.ego_predicted_position = transform_point(predicted_position, scene.ego.world_from_ego)
        self.ego_predicted_yaw = scene.ego.yaw + predicted_yaw


class CollisionMetric(Metric):
    name = "collision"
    
    def evaluate(self, scene: Scene, predicted_position: np.ndarray, predicted_yaw: float) -> float:
        super().evaluate(scene, predicted_position, predicted_yaw)
        
        ego_bounding_box = _get_bounding_box(self.ego_predicted_position, self.ego_predicted_yaw, scene.ego.extent)
        ego_area = scene.ego.length * scene.ego.width
        
        min_score = 0.0
        
        for _, agent in scene.agents.items():
            
            agent_bounding_box = _get_bounding_box(agent.position, agent.yaw, agent.extent)
            intersection = agent_bounding_box.intersection(ego_bounding_box)
            score = -intersection.area / ego_area
            
            if score < min_score:
                min_score = score

        return min_score


class SafeDistanceMetric(Metric):
    name = "safe_distance"
    
    def evaluate(self, scene: Scene, predicted_position: np.ndarray, predicted_yaw: float) -> float:
        super().evaluate(scene, predicted_position, predicted_yaw)
        
        return 0.0
    