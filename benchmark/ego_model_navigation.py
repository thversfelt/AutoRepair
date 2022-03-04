from collections import deque
from os import close
from typing import Dict
from l5kit.geometry.transform import transform_point, transform_points
import numpy as np
import torch
import torch.nn as nn
from benchmark.agent import EgoAgent
from benchmark.custom_map_api import CustomMapAPI
import random

from benchmark.ego_model_perception import EgoModelPerception
from benchmark.scene import Scene


class EgoModelNavigation():

    def process(self, scene: Scene):
        steer = 0.0
        closest_midpoint = None
            
        for lane_id in scene.ego.route:
            # Get the closest lane midpoints for the ego's current lane.
            lane_closest_midpoints = scene.map.get_closest_lane_midpoints(scene.ego.position, lane_id)
            lane_closest_midpoints = transform_points(lane_closest_midpoints, scene.ego.ego_from_world)
            lane_closest_midpoints = lane_closest_midpoints[lane_closest_midpoints[:,0] > 0]
            
            if len(lane_closest_midpoints) == 0:
                continue
            else:
                closest_midpoint = lane_closest_midpoints[0]
                break

        if closest_midpoint is not None:
            # Steer input is proportional to the y-coordinate of the closest midpoint.
            steer = 0.5 * closest_midpoint[1]
        
        return steer
