from autotest.model.context.scene import Scene


class Metric:
    name = ""
    
    def evaluate(self, scene: Scene) -> float:
        pass


class CollisionMetric(Metric):
    name = "collision"
    
    def evaluate(self, scene: Scene) -> float:
        return 0.0


class SafeDistanceMetric(Metric):
    name = "safe_distance"
    
    def evaluate(self, scene: Scene) -> float:
        return 0.0
    