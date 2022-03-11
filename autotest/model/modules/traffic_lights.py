from autotest.model.context.scene import Scene


class TrafficLights():

    UNKNOWN = 0, 
    GREEN = 1
    YELLOW = 2
    RED = 3
    NONE = 4

    ACCELERATION = 1.0 # [m/s^2]
    DECELLERATION = -8.0 # [m/s^2]

    def process(self, scene: Scene) -> float:
        if scene.ego.traffic_light == self.RED and scene.ego.speed > 0.0:
            return self.DECELLERATION
        else:
            return self.ACCELERATION
