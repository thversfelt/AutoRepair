import random
from l5kit.geometry.transform import transform_point, transform_points
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
from benchmark.custom_map_api import CustomMapAPI
from benchmark.scene import Scene


class EgoModelPerception():
    def __init__(self, map: CustomMapAPI):
        self.map = map
        self.scenes: Dict[Scene] = {}

    def process(self, data_batch: Dict[str, torch.Tensor]):
        scene_ids = data_batch['scene_index'].cpu().numpy()
        for scene_index, scene_id in enumerate(scene_ids):
            if scene_index not in self.scenes:
                self.scenes[scene_index] = Scene(scene_index, scene_id, self.map)
            scene = self.scenes[scene_index]
            scene.update(data_batch)

    def update_traffic_lights(self, data_batch: Dict[str, torch.Tensor]):
        num_of_scenes = len(data_batch['scene_index'])
        
        for scene_idx in range(num_of_scenes):
            # Initialize this scene's dict which stores the traffic light status of the lanes on the ego's route.
            self.traffic_lights[scene_idx] = {}
            
            # Decode the list of active traffic light faces id's in this scene.
            traffic_light_faces_ids = data_batch["traffic_light_faces_ids"][scene_idx].cpu().numpy()
            traffic_light_faces_ids = [self.map_api.int_as_id(face_id) for face_id in traffic_light_faces_ids]
            
            # Retrieve the color of these traffic light faces.
            traffic_light_faces_colors = data_batch["traffic_light_faces_colors"][scene_idx].cpu().numpy()
            
            for lane_id in self.ego_route[scene_idx]:
                # Get the traffic control id's for this lane.
                traffic_control_ids = self.map_api.get_lane_traffic_control_ids(lane_id)
                
                for idx, traffic_control_id in enumerate(traffic_control_ids):
                    # If the traffic control id is found in the list of active traffic lights, save this lane's traffic
                    # light color (red, yellow or green).
                    if traffic_control_id in traffic_light_faces_ids:
                        self.traffic_lights[scene_idx][lane_id] = traffic_light_faces_colors[idx]
