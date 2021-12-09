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
        else:
            return lane_id

    def get_closest_lane(self, position: np.ndarray) -> str:
        """Gets the closest lane of the given position.

        Args:
            position (np.ndarray): The position to get the closest lane of.

        Returns:
            str: The lane id of the closest lane to the given position.
        """
        max_lane_distance = self.lane_cfg_params["max_retrieval_distance_m"]  # Only consider lanes within this distance to the given position.

        # filter first by bounds and then by distance, so that we always take the closest lanes
        lanes_ids = self.bounds_info["lanes"]["ids"]
        lanes_bounds = self.bounds_info["lanes"]["bounds"]
        lanes_indices = indices_in_bounds(position, lanes_bounds, max_lane_distance)
        if len(lanes_indices) == 0:
            return None  # There are no lanes close to the given position, return None.
        
        lanes_distances = []
        for lane_idx in lanes_indices:
            lane_id = lanes_ids[lane_idx]
            closest_midpoints = self.get_closest_lane_midpoints(position, lane_id)  # Determine closest lane midpoints to the given position.
            closest_midpoint = closest_midpoints[0]  # The closest midpoint is the first element of the sorted list of lane midpoints.
            closest_midpoint_distance = np.linalg.norm(closest_midpoint - position)  # Determine distance from the closest midpoint to the given position.
            lanes_distances.append(closest_midpoint_distance)  # Assign the lane distance to be the closest midpoint to the given position.
        lanes_indices = lanes_indices[np.argsort(lanes_distances)]  # Sort the lane indices by lane distance, ascending.
        
        closest_lane_idx = lanes_indices[0]
        closest_lane_id = lanes_ids[closest_lane_idx]
        return closest_lane_id

    def get_closest_lane_midpoints(self, position: np.ndarray, lane_id: str) -> np.ndarray:
        """Gets a sorted list (ascending) of midpoints of the lane, defined by the lane id, that are closest to the given position.

        Args:
            position (np.ndarray): The position to get the closest midpoints to.
            lane_id (str): The id of the lane to get the closest midpoints of.

        Returns:
            np.ndarray: A sorted list (ascending) of midpoints, that are closest to the given position.
        """
        max_lane_points = self.lane_cfg_params["max_points_per_lane"]  # Maximum amounts of points to represent lane.
        interpolation_method = InterpolationMethod.INTER_ENSURE_LEN  # Split lane in a fixed number of points.
        lane = self.get_lane_as_interpolation(lane_id, max_lane_points, interpolation_method)
        midpoints = lane["xyz_midlane"][:, :2]  # Retrieve the lane's midpoints.
        midpoints_distance = np.linalg.norm(midpoints - position, axis=-1)  # Determine the distance between each midpoint and the given position.
        closest_midpoints = midpoints[np.argsort(midpoints_distance)]  # Sort the midpoints by midpoint distance to the given position, ascending.
        return closest_midpoints
        
