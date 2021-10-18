import sys
import numpy as np
from highway_env.vehicle.kinematics import Vehicle


def get_front_vehicle(vehicle: Vehicle) -> Vehicle:
    front_vehicle, _ = vehicle.road.neighbour_vehicles(vehicle)
    return front_vehicle


def get_front_gap(vehicle: Vehicle) -> float:
    front_vehicle = get_front_vehicle(vehicle)
    if front_vehicle is None:
        return sys.maxsize
    else:
        return np.linalg.norm(front_vehicle.position - vehicle.position)
