from gym.envs.registration import register
from highway_env.envs import AbstractEnv, Action
from highway_env.road.road import RoadNetwork, Road
from highway_env.vehicle.kinematics import Vehicle
from testsuite.objectives import scores


class FollowEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "ContinuousAction",
            },
            "duration": 50,  # [steps]
            "policy_frequency": 1,  # [steps per policy evaluation]
            "speed_limit": 30.0  # [m/s]
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(lanes=1),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"]
        )

    def _create_vehicles(self) -> None:
        controlled_vehicle = Vehicle(
            road=self.road,
            position=self.road.network.get_lane(("0", "1", 0)).position(0, 0),
            speed=20.0
        )
        leading_vehicle = Vehicle(
            road=self.road,
            position=self.road.network.get_lane(("0", "1", 0)).position(50, 0),
            speed=15.0
        )
        self.road.vehicles = [controlled_vehicle, leading_vehicle]
        self.controlled_vehicles = [controlled_vehicle]

    def _reward(self, action: Action) -> [float]:
        return scores(self)

    def _is_terminal(self) -> bool:
        return self.vehicle.crashed or self.steps >= self.config["duration"]

    def _cost(self, action: Action) -> float:
        pass


register(
    id='follow-env-v0',
    entry_point='testsuite.envs:FollowEnv',
)
