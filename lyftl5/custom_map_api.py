from typing import List
import numpy as np
import math
from l5kit.configs.config import load_metadata
from l5kit.data import MapAPI, DataManager
from l5kit.data.map_api import InterpolationMethod
from l5kit.data.proto.road_network_pb2 import Lane, MapElement
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds
from queue import PriorityQueue
from collections import deque


class CustomMapAPI(MapAPI):
    def __init__(self, cfg: dict, dm: DataManager):
        dataset_meta_key = cfg["raster_params"]["dataset_meta_key"]
        dataset_meta = load_metadata(dm.require(dataset_meta_key))
        world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
        protobuf_map_path = dm.require(cfg["raster_params"]["semantic_map_key"])
        super().__init__(protobuf_map_path, world_to_ecef)

        self.lane_cfg_params = cfg["data_generation_params"]["lane_params"]

    def get_closest_lanes_ids(self, position: np.ndarray) -> List[str]:
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
            lane_id = lanes_ids[lane_idx]  # Get the id of the line index (id != index, id is a string, index and integer index of a list).
            closest_midpoints = self.get_closest_lane_midpoints(position, lane_id)  # Determine closest lane midpoints to the given position.
            closest_midpoint = closest_midpoints[0]  # The closest midpoint is the first element of the sorted list of lane midpoints.
            closest_midpoint_distance = np.linalg.norm(closest_midpoint - position)  # Determine distance from the closest midpoint to the given position.
            lanes_distances.append(closest_midpoint_distance)  # Assign the lane distance to be the closest midpoint to the given position.
        lanes_indices = lanes_indices[np.argsort(lanes_distances)]  # Sort the lane indices by lane distance, ascending.
        
        closest_lanes_ids = np.take(lanes_ids, lanes_indices)  # Get the ids of the sorted list of lane indices.
        return closest_lanes_ids

    def get_closest_lane_midpoints(self, position: np.ndarray, lane_id: str) -> np.ndarray:
        """Gets a sorted list (ascending) of midpoints of the lane, defined by the lane id, that are closest to the given position.

        Args:
            position (np.ndarray): The position to get the closest midpoints to.
            lane_id (str): The id of the lane to get the closest midpoints of.

        Returns:
            np.ndarray: A sorted list (ascending) of midpoints, that are closest to the given position.
        """
        #max_lane_points = self.lane_cfg_params["max_points_per_lane"]  # Maximum amounts of points to represent lane.
        interpolation_method = InterpolationMethod.INTER_METER  # Split lane in a fixed number of points.
        lane = self.get_lane_as_interpolation(lane_id, 1.0, interpolation_method)
        midpoints = lane["xyz_midlane"][:, :2]  # Retrieve the lane's midpoints.
        midpoints_distance = np.linalg.norm(midpoints - position, axis=-1)  # Determine the distance between each midpoint and the given position.
        closest_midpoints = midpoints[np.argsort(midpoints_distance)]  # Sort the midpoints by midpoint distance to the given position, ascending.
        return closest_midpoints
    
    def get_shortest_route(self, start_position: np.ndarray, end_position: np.ndarray) -> deque:
        """Implements Dijkstra's Algorithm (https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm).

        Args:
            start_position (np.ndarray): Start position coordinates (x, y) in world reference system.
            end_position (np.ndarray): End position coordinates (x, y) in world reference system.

        Returns:
            List[str]: The route as a list of lane id's.
        """
        start_lane_id = self.get_closest_lanes_ids(start_position)[0]
        end_lane_id = self.get_closest_lanes_ids(end_position)[0]
        
        # A priority queue with tuples (distance, (lane_id, parent_lane_id)), which contains the distances from the 
        # start lane to the lane represented by lane_id, through the parent lane represented by parent_lane_id.
        explored_lanes_ids = PriorityQueue() 
        visited_lanes_ids = {}
        explored_lanes_ids.put((0, (start_lane_id, start_lane_id)))
        
        # While the list of visisted lanes doesn't contain any end lane, keep searching.
        while end_lane_id not in visited_lanes_ids:
            # Get the unvisited lane that has the smallest distance to the start lane from the priority queue.
            explored_lane = explored_lanes_ids.get()
            explored_lane_distance = explored_lane[0]
            explored_lane_id = explored_lane[1][0]
            explored_lane_parent_id = explored_lane[1][1]
            
            # Add the explored lane to the set of visited lanes.
            if explored_lane_id not in visited_lanes_ids:
                visited_lanes_ids[explored_lane_id] = explored_lane_parent_id
            
            # Explore all the connected lanes that have not been visited yet, and append the distance from this lane to 
            # that lane to the recorded distance from the start lane. 
            for lane_id in self.get_connected_lanes_ids(explored_lane_id):
                if lane_id not in visited_lanes_ids:
                    lane_length = self.get_approx_lane_length(lane_id)
                    explored_lanes_ids.put((explored_lane_distance + lane_length, (lane_id, explored_lane_id)))
        
        # Backtrack the output to get the shortest route.
        shortest_route = deque()
        lane_id = end_lane_id
        while start_lane_id not in shortest_route:
            shortest_route.appendleft(lane_id)  # Add the lane id to the shortest route.
            lane_id = visited_lanes_ids[lane_id]  # Set the lane id as the parent of the current lane.
        return shortest_route
    
    def get_element(self, element_id: str) -> MapElement:
        element_idx = self.ids_to_el[element_id]
        element = self.elements[element_idx]
        return element
    
    def get_lane(self, lane_id: str) -> Lane:
        element = self.get_element(lane_id)
        return element.element.lane

    def get_connected_lanes_ids(self, lane_id: str) -> List[str]:
        ahead_lanes_ids = self.get_ahead_lanes_ids(lane_id)
        change_lanes_ids = self.get_change_lanes_ids(lane_id)
        return ahead_lanes_ids + change_lanes_ids

    def get_ahead_lanes_ids(self, lane_id: str) -> List[str]:
        lane = self.get_lane(lane_id)
        ahead_lanes_ids = []
        for ahead_lane in lane.lanes_ahead:
            ahead_lane_id = self.id_as_str(ahead_lane)
            if bool(ahead_lane_id):
                ahead_lanes_ids.append(ahead_lane_id)
        return ahead_lanes_ids

    def get_change_lanes_ids(self, lane_id: str) -> List[str]:
        lane = self.get_lane(lane_id)
        change_lanes_ids = []
        change_left_lane_id = self.id_as_str(lane.adjacent_lane_change_left)
        change_right_lane_id = self.id_as_str(lane.adjacent_lane_change_right)
        if bool(change_left_lane_id):
            change_lanes_ids.append(change_left_lane_id)
        if bool(change_right_lane_id):
            change_lanes_ids.append(change_right_lane_id)
        return change_lanes_ids

    def get_approx_lane_length(self, lane_id: str) -> float:
        lane = self.get_lane_coords(lane_id)
        
        # Get the lane bounds.
        x_min = min(np.min(lane["xyz_left"][:, 0]), np.min(lane["xyz_right"][:, 0]))
        y_min = min(np.min(lane["xyz_left"][:, 1]), np.min(lane["xyz_right"][:, 1]))
        x_max = max(np.max(lane["xyz_left"][:, 0]), np.max(lane["xyz_right"][:, 0]))
        y_max = max(np.max(lane["xyz_left"][:, 1]), np.max(lane["xyz_right"][:, 1]))
        
        # Approximate the lane length as the lane bounds diagional length.
        x_length = abs(x_max - x_min)
        y_length = abs(y_max - y_min)
        lane_length = math.hypot(x_length, y_length)
        return lane_length
    
    def get_lane_progress(self, position: np.ndarray, lane_id: str) -> float:
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
        progress = np.linspace(0, 1, len(midpoints))
        midpoints_distance = np.linalg.norm(midpoints - position, axis=-1)  # Determine the distance between each midpoint and the given position.
        progress = progress[np.argsort(midpoints_distance)]  # Sort the midpoints by midpoint distance to the given position, ascending.
        return progress[0]

    def get_lane_speed_limit(self, lane_id: str) -> float:
        return 14.0