from typing import List
import numpy as np
from l5kit.configs.config import load_metadata
from l5kit.data import MapAPI, DataManager
from l5kit.data.map_api import InterpolationMethod
from l5kit.data.proto.road_network_pb2 import Lane
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds
from l5kit.geometry.transform import transform_points


class CustomMapAPI(MapAPI):
    def __init__(self, cfg: dict, dm: DataManager):
        dataset_meta_key = cfg["raster_params"]["dataset_meta_key"]
        dataset_meta = load_metadata(dm.require(dataset_meta_key))
        world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
        protobuf_map_path = dm.require(cfg["raster_params"]["semantic_map_key"])
        super().__init__(protobuf_map_path, world_to_ecef)

        self.lane_cfg_params = cfg["data_generation_params"]["lane_params"]

    def get_next_lane(self, lane_id: str) -> str:
        lane_idx = self.ids_to_el[lane_id]
        element = self.elements[lane_idx]
        lane = element.element.lane
        lanes_ahead = lane.lanes_ahead
        if len(lanes_ahead) > 0:
            lane_ahead = np.random.choice(lanes_ahead)
            return self.id_as_str(lane_ahead)

    def get_closest_lane(self, position: np.ndarray) -> str:
        """Gets the closest lane of the given position.

        Args:
            position (np.ndarray): The position to get the closest lane of.

        Returns:
            str: The lane id.
        """
        max_lane_distance = self.lane_cfg_params["max_retrieval_distance_m"]  # Only look at lanes within this distance.
        max_lane_points = self.lane_cfg_params["max_points_per_lane"]  # Maximum amounts of points to represent lane.
        interpolation_method = InterpolationMethod.INTER_ENSURE_LEN  # Split lane in a fixed number of points

        # filter first by bounds and then by distance, so that we always take the closest lanes
        lanes_ids = self.bounds_info["lanes"]["ids"]
        lanes_bounds = self.bounds_info["lanes"]["bounds"]
        lanes_indices = indices_in_bounds(position, lanes_bounds, max_lane_distance)
        distances = []
        for lane_idx in lanes_indices:
            lane_id = lanes_ids[lane_idx]
            lane = self.get_lane_as_interpolation(lane_id, max_lane_points, interpolation_method)
            midlanes = lane["xyz_midlane"][:, :2]
            midlanes_distance = np.linalg.norm(midlanes - position, axis=-1)
            distances.append(np.min(midlanes_distance))
        lanes_indices = lanes_indices[np.argsort(distances)]
        closest_lane_idx = lanes_indices[0]
        closest_lane_id = lanes_ids[closest_lane_idx]
        return closest_lane_id

    def get_closest_lane_midpoint(self, lane_id: str, transf_matrix: np.ndarray) -> np.ndarray:
        """[summary]

        Args:
            position (np.ndarray): [description]
            lane_id (str): [description]

        Returns:
            np.ndarray: [description]
        """
        max_lane_points = self.lane_cfg_params["max_points_per_lane"]  # Maximum amounts of points to represent lane.
        interpolation_method = InterpolationMethod.INTER_ENSURE_LEN  # Split lane in a fixed number of points
        
        lane = self.get_lane_as_interpolation(lane_id, max_lane_points, interpolation_method)
        midlane = lane["xyz_midlane"][:, :2]
        midlane = transform_points(midlane, transf_matrix)
        
        next_midpoint = None
        next_midpoint_distance = None
        for midpoint in midlane:
            if midpoint[0] <= 0:  # X-coordinate is equal or less than 0 in the agent's reference system, so is behind.
                continue  # Filter points that are behind the agent in its reference system.
            midpoint_distance = np.linalg.norm(midpoint)
            if next_midpoint is None:
                next_midpoint = midpoint
                next_midpoint_distance = midpoint_distance
            elif midpoint_distance < next_midpoint_distance:
                next_midpoint = midpoint
                next_midpoint_distance = midpoint_distance
        
        return next_midpoint
