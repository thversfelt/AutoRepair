import numpy as np

from rtree import index
from shapely.geometry import box, Polygon, Point
from typing import List, Tuple
from l5kit.configs.config import load_metadata
from l5kit.data import MapAPI, DataManager
from l5kit.data.map_api import InterpolationMethod, ENCODING
from l5kit.data.proto.road_network_pb2 import Lane, MapElement, RoadNetworkSegment, Junction
from collections import deque


class CustomMapAPI(MapAPI):
    def __init__(self, cfg: dict, dm: DataManager):
        dataset_meta_key = cfg["raster_params"]["dataset_meta_key"]
        dataset_meta = load_metadata(dm.require(dataset_meta_key))
        world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
        protobuf_map_path = dm.require(cfg["raster_params"]["semantic_map_key"])
        super().__init__(protobuf_map_path, world_to_ecef)
        
        self.lanes_ids = self.get_lanes_ids()
        self.lanes_spatial_index = self.get_lanes_spatial_index()
        
        self.segments_ids = self.get_segments_ids()
        self.junctions_ids = self.get_junctions_ids()

    def get_lanes_spatial_index(self):
        # https://rtree.readthedocs.io/en/latest/index.html
        lanes_spatial_index = index.Index()
        
        for lane_idx, lane_id in enumerate(self.lanes_ids):
            (x_min, y_min, x_max, y_max) = self.get_lane_extent(lane_id)
            lanes_spatial_index.insert(id=lane_idx, coordinates=(x_min, y_min, x_max, y_max), obj=lane_id)
        
        return lanes_spatial_index

    def get_lanes_ids(self) -> List[str]:
        lanes_ids = []
        
        for element in self.elements:
            if self.is_lane(element):
                lane_id = self.id_as_str(element.id)
                lanes_ids.append(lane_id)
                
        return lanes_ids

    def get_segments_ids(self) -> List[str]:
        segments_ids = []
        
        for element in self.elements:
            if self.is_segment(element):
                segment_id = self.id_as_str(element.id)
                segments_ids.append(segment_id)
                
        return segments_ids
    
    def get_junctions_ids(self) -> List[str]:
        junction_ids = []
        
        for element in self.elements:
            if self.is_junction(element):
                junction_id = self.id_as_str(element.id)
                junction_ids.append(junction_id)
                
        return junction_ids
    
    def is_segment(self, element: MapElement) -> bool:
        return bool(element.element.HasField("segment"))

    def is_junction(self, element: MapElement) -> bool:
        return bool(element.element.HasField("junction"))

    def get_route(self, trajectory: np.ndarray) -> List[str]:
        # Ensure the trajectory has a single position.
        if len(trajectory) == 0:
            return None

        # Define the sets of visited and candidate lanes.
        candidate_lanes_ids = {}
        visited_lanes_ids = {}
        
        # Get the lanes at the initial position of the trajectory.
        initial_position = trajectory[0]
        initial_lanes_ids = self.get_lanes_ids_at(initial_position)
        
        # Get the lanes at the last position of the trajectory.
        final_position = trajectory[-1]
        final_lanes_ids = self.get_lanes_ids_at(final_position)
        
        # Set the initial lanes as the candidate lanes, with "None" as their parent lane id, indicating that these 
        # lanes do not have any parent lanes.
        for initial_lane_id in initial_lanes_ids:
            candidate_lanes_ids[initial_lane_id] = None

        last_visited_lane_id = None

        # Then, for each position of the trajectory:
        for position in trajectory:
            # Get the candidate lanes at the current position.
            current_lanes_ids = []
            for lane_id in candidate_lanes_ids:
                if self.in_lane(position, lane_id):
                    current_lanes_ids.append(lane_id)

            # Visit these candidate lanes.
            for lane_id in current_lanes_ids:
                if lane_id in visited_lanes_ids: 
                    continue
                
                visited_lanes_ids[lane_id] = candidate_lanes_ids[lane_id]
                last_visited_lane_id = lane_id

            # If at least one new lane has been visited, reset the set of candidate lanes.
            if len(current_lanes_ids) > 0:
                candidate_lanes_ids = {}
                
                # Then, add all connected lanes of the newly visisted lanes to the set of candidate lanes. 
                for lane_id in current_lanes_ids:
                    for connected_lane_id in self.get_connected_lanes_ids(lane_id):
                        if connected_lane_id in candidate_lanes_ids: continue
                        candidate_lanes_ids[connected_lane_id] = lane_id
            else:
                # A position is off-road when it is not in any of the candidate or visited lanes.
                off_road = True
                for visited_lane_id in visited_lanes_ids.keys():
                    if self.in_lane(position, visited_lane_id):
                        off_road = False
                
                # If the position is off-road, expand the set of candidate lanes. Note that off-road means that it is 
                # not on any "known" lanes (candidate or visited), so add any potentially "unknown" lanes at this 
                # position to the set of candidate lanes.
                if off_road and last_visited_lane_id is not None:
                    current_lanes_ids = self.get_lanes_ids_at(position)
                    for current_lane_id in current_lanes_ids:
                        candidate_lanes_ids[current_lane_id] = last_visited_lane_id

        # Backtrack the visited lanes, starting from the lanes at the final position, to obtain a feasible route.
        route = None
        for final_lane_id in final_lanes_ids:
            if final_lane_id not in visited_lanes_ids: 
                continue
            
            route = deque()
            lane_id = final_lane_id
            while not any(x in initial_lanes_ids for x in route):
                route.appendleft(lane_id)
                lane_id = visited_lanes_ids[lane_id]
            break
        
        return route

    def get_lanes_ids_at(self, position: np.ndarray) -> List[str]:
        """Gets the lanes at the given position.

        Args:
            position (np.ndarray): The position to get the lanes at.

        Returns:
            str: The lane ids of the lanes at the given position.
        """
        x = position[0]
        y = position[1]
        
        lanes_ids = [n.object for n in self.lanes_spatial_index.intersection((x, y, x, y), objects=True)]

        return lanes_ids

    def get_closest_lane_midpoints(self, position: np.ndarray, lane_id: str) -> np.ndarray:
        """Gets a sorted list (ascending) of midpoints of the lane, defined by the lane id, that are closest to the given position.

        Args:
            position (np.ndarray): The position to get the closest midpoints to.
            lane_id (str): The id of the lane to get the closest midpoints of.

        Returns:
            np.ndarray: A sorted list (ascending) of midpoints, that are closest to the given position.
        """
        interpolation_method = InterpolationMethod.INTER_METER  # Split lane in a fixed number of points.
        lane = self.get_lane_as_interpolation(lane_id, 1.0, interpolation_method)
        midpoints = lane["xyz_midlane"][:, :2]  # Retrieve the lane's midpoints.
        midpoints_distance = np.linalg.norm(midpoints - position, axis=-1)  # Determine the distance between each midpoint and the given position.
        closest_midpoints = midpoints[np.argsort(midpoints_distance)]  # Sort the midpoints by midpoint distance to the given position, ascending.
        return closest_midpoints
    
    def get_element(self, element_id: str) -> MapElement:
        element_idx = self.ids_to_el[element_id]
        element = self.elements[element_idx]
        return element
    
    def get_lane(self, lane_id: str) -> Lane:
        element = self.get_element(lane_id)
        return element.element.lane

    def get_segment(self, segment_id: str) -> RoadNetworkSegment:
        element = self.get_element(segment_id)
        return element.element.segment
    
    def get_junction(self, junction_id: str) -> Junction:
        element = self.get_element(junction_id)
        return element.element.junction

    def get_segment_speed_limit(self, segment_id: str) -> float:
        segment = self.get_segment(segment_id)
        return segment.speed_limit_meters_per_second

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

    def in_lane(self, position: np.ndarray, lane_id: str) -> bool:
        lane_polygon = self.get_lane_polygon(lane_id)
        point = Point(position[0], position[1])
        
        if lane_polygon.contains(point):
            return True
        else:
            return False

    def get_lane_extent(self, lane_id: str) -> Tuple:
        lane_coordinates = self.get_lane_coords(lane_id)
        
        x_min = min(np.min(lane_coordinates["xyz_left"][:, 0]), np.min(lane_coordinates["xyz_right"][:, 0]))
        y_min = min(np.min(lane_coordinates["xyz_left"][:, 1]), np.min(lane_coordinates["xyz_right"][:, 1]))
        x_max = max(np.max(lane_coordinates["xyz_left"][:, 0]), np.max(lane_coordinates["xyz_right"][:, 0]))
        y_max = max(np.max(lane_coordinates["xyz_left"][:, 1]), np.max(lane_coordinates["xyz_right"][:, 1]))
        
        return x_min, y_min, x_max, y_max

    def get_lane_bounds(self, lane_id: str) -> Polygon:
        x_min, y_min, x_max, y_max = self.get_lane_extent(lane_id)
        return box(x_min, y_min, x_max, y_max)

    def get_lane_polygon(self, lane_id: str) -> Polygon:
        lane_coordinates = self.get_lane_coords(lane_id)
        
        lane_left_boundary = lane_coordinates["xyz_left"][:, :2]
        lane_right_boundary = lane_coordinates["xyz_right"][:, :2]
        lane_vertices = np.concatenate([lane_left_boundary, lane_right_boundary[::-1], [lane_left_boundary[0]]])
        
        return Polygon(lane_vertices)

    def get_lane_speed_limit(self, lane_id: str) -> float:
        lane = self.get_lane(lane_id)
        segment_or_junction_id = self.id_as_str(lane.parent_segment_or_junction)
        
        if segment_or_junction_id in self.segments_ids:
            return self.get_segment_speed_limit(segment_or_junction_id)

    def is_lane_in_junction(self, lane_id: str) -> bool:
        lane = self.get_lane(lane_id)
        segment_or_junction_id = self.id_as_str(lane.parent_segment_or_junction)
        
        if segment_or_junction_id in self.junctions_ids:
            return True
        else:
            return False

    def id_as_int(self, id: str) -> np.int32:
        # Add padding (whitespace) to ids with less than 4 charachters.
        padded_id = id.ljust(4)
        
        # Encode the (padded) id to bytes.
        id_bytes = padded_id.encode(ENCODING)
        
        # Convert the bytes to a 32-bit integer.
        return np.frombuffer(id_bytes, dtype=np.int32)

    def int_as_id(self, id: np.int32) -> str:
        # Convert the 32-bit integer to bytes.
        id_bytes = np.frombuffer(id, dtype=np.int8).tobytes()
        
        # Decode the (padded) id to a string.
        padded_id = id_bytes.decode(ENCODING)
        
        # Remove any padding (whitespace) from the id.
        return padded_id.strip()
