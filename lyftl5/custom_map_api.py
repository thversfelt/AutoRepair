from typing import List
import numpy as np
from l5kit.configs.config import load_metadata
from l5kit.data import MapAPI, DataManager
from l5kit.data.map_api import InterpolationMethod
from l5kit.data.proto.road_network_pb2 import Lane
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds


class CustomMapAPI(MapAPI):
    def __init__(self, cfg: dict, dm: DataManager):
        dataset_meta_key = cfg["raster_params"]["dataset_meta_key"]
        dataset_meta = load_metadata(dm.require(dataset_meta_key))
        world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
        protobuf_map_path = dm.require(cfg["raster_params"]["semantic_map_key"])
        super().__init__(protobuf_map_path, world_to_ecef)

        self.lane_cfg_params = cfg["data_generation_params"]["lane_params"]

    def get_closest_lane(self, position: np.ndarray) -> Lane:
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
            lane_dist = np.linalg.norm(lane["xyz_midlane"][:, :2] - position, axis=-1)
            distances.append(np.min(lane_dist))
        lanes_indices = lanes_indices[np.argsort(distances)]
        closest_lane_idx = lanes_indices[0]
        closest_lane_id = lanes_ids[closest_lane_idx]
        closest_lane_element = self.elements[self.ids_to_el[closest_lane_id]]
        return closest_lane_element.element.lane
