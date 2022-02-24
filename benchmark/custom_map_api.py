from typing import List
import numpy as np
from l5kit.configs.config import load_metadata
from l5kit.data import MapAPI, DataManager
from l5kit.data.map_api import InterpolationMethod, ENCODING
from l5kit.data.proto.road_network_pb2 import Lane, MapElement
from collections import deque


class CustomMapAPI(MapAPI):
    def __init__(self, cfg: dict, dm: DataManager):
        dataset_meta_key = cfg["raster_params"]["dataset_meta_key"]
        dataset_meta = load_metadata(dm.require(dataset_meta_key))
        world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
        protobuf_map_path = dm.require(cfg["raster_params"]["semantic_map_key"])
        super().__init__(protobuf_map_path, world_to_ecef)

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

        # Then, for each position of the trajectory:
        for position in trajectory:
            # Get the candidate lanes at the current position.
            current_lanes_ids = []
            for lane_id in candidate_lanes_ids:
                if self.in_lane(position, lane_id):
                    current_lanes_ids.append(lane_id)
            
            # Visit these candidate lanes and their (potentially thin, or skipped) parent.
            for lane_id in current_lanes_ids:
                if lane_id in visited_lanes_ids: continue
                visited_lanes_ids[lane_id] = candidate_lanes_ids[lane_id]
                parent_lane_id = candidate_lanes_ids[lane_id]
                if parent_lane_id is None or parent_lane_id in visited_lanes_ids: continue
                visited_lanes_ids[parent_lane_id] = candidate_lanes_ids[parent_lane_id]

            # If at least one new lane has been visited, reset the set of candidate lanes.
            if len(current_lanes_ids) > 0:
                candidate_lanes_ids = {}
                
                # Then, add all connected lanes (depth of 2, in case of very thin or skipped lanes) of the newly 
                # visisted lanes to the set of candidate lanes. 
                for lane_id in current_lanes_ids:
                    for connected_lane_id in self.get_connected_lanes_ids(lane_id):
                        if connected_lane_id in candidate_lanes_ids: continue
                        candidate_lanes_ids[connected_lane_id] = lane_id
                        for secondary_connected_lane_id in self.get_connected_lanes_ids(connected_lane_id):
                            if secondary_connected_lane_id in candidate_lanes_ids: continue
                            candidate_lanes_ids[secondary_connected_lane_id] = connected_lane_id
        
        # Backtrack the visited lanes, starting from the lanes at the final position, to obtain a feasible route.
        route = None
        for final_lane_id in final_lanes_ids:
            if final_lane_id not in visited_lanes_ids: continue
            route = deque()
            lane_id = final_lane_id
            while not any(x in initial_lanes_ids for x in route):
                route.appendleft(lane_id)
                lane_id = visited_lanes_ids[lane_id]
            break
        
        # import matplotlib.pyplot as plt
        # plt.figure()
        # for visited_lane_id in visited_lanes_ids:
        #     lane_polygon = self.get_lane_polygon(visited_lane_id)
        #     xs, ys = zip(*lane_polygon) #create lists of x and y values
        #     plt.plot(xs,ys, color='green')
        # for unvisited_lane_id in unvisited_lanes_ids:
        #     lane_polygon = self.get_lane_polygon(unvisited_lane_id)
        #     xs, ys = zip(*lane_polygon) #create lists of x and y values
        #     plt.plot(xs,ys, color='red')
        # for lane_id in route:
        #     lane_polygon = self.get_lane_polygon(lane_id)
        #     xs, ys = zip(*lane_polygon) #create lists of x and y values
        #     plt.plot(xs,ys, color='blue')
        # for position in trajectory:
        #     x = position[0]
        #     y = position[1]
        #     plt.scatter(x, y, color='black', s=1)
        # plt.show()
        
        return route
    

    def get_lanes_ids_at(self, position: np.ndarray) -> List[str]:
        """Gets the lanes at the given position.

        Args:
            position (np.ndarray): The position to get the lanes at.

        Returns:
            str: The lane ids of the lanes at the given position.
        """
        lanes_ids = []
        for element in self.elements:
            if not self.is_lane(element): continue
            lane_id = self.id_as_str(element.id)
            if not self.in_lane(position, lane_id): continue
            lanes_ids.append(lane_id)
        return lanes_ids

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

    def in_lane(self, position: np.ndarray, lane_id: str) -> bool:
        if not self.in_lane_bounds(position, lane_id): return False
        if not self.in_lane_exact(position, lane_id): return False
        return True

    def in_lane_bounds(self, position: np.ndarray, lane_id: str) -> bool:
        lane_coords = self.get_lane_coords(lane_id)
        
        # Get the lane bounds.
        x_min = min(np.min(lane_coords["xyz_left"][:, 0]), np.min(lane_coords["xyz_right"][:, 0]))
        y_min = min(np.min(lane_coords["xyz_left"][:, 1]), np.min(lane_coords["xyz_right"][:, 1]))
        x_max = max(np.max(lane_coords["xyz_left"][:, 0]), np.max(lane_coords["xyz_right"][:, 0]))
        y_max = max(np.max(lane_coords["xyz_left"][:, 1]), np.max(lane_coords["xyz_right"][:, 1]))

        bounds = np.asarray([[x_min, y_min], [x_max, y_max]])
        return self.in_bounds(position, bounds)

    def in_lane_exact(self, position: np.ndarray, lane_id: str) -> bool:
        x = position[0]
        y = position[1]
        lane_coords = self.get_lane_coords(lane_id)
        lane_left_boundary = lane_coords["xyz_left"][:, :2]
        lane_right_boundary = lane_coords["xyz_right"][:, :2]
        poly = np.concatenate([lane_left_boundary, lane_right_boundary[::-1], [lane_left_boundary[0]]])

        n = len(poly)
        inside = False
        p1x,p1y = poly[0]
        for i in range(n+1):
            p2x,p2y = poly[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x,p1y = p2x,p2y

        return inside

    def get_lane_polygon(self, lane_id: str) -> np.ndarray:
        lane_coords = self.get_lane_coords(lane_id)
        lane_left_boundary = lane_coords["xyz_left"][:, :2]
        lane_right_boundary = lane_coords["xyz_right"][:, :2]
        lane_polygon = np.concatenate([lane_left_boundary, lane_right_boundary[::-1], [lane_left_boundary[0]]])
        return lane_polygon

    def in_bounds(self, position: np.ndarray, bounds: np.ndarray) -> bool:
        x = position[0]
        y = position[1]
        
        # Get the lane bounds.
        x_min = bounds[0][0]
        y_min = bounds[0][1]
        x_max = bounds[1][0]
        y_max = bounds[1][1]
        
        x_in = x > x_min and x < x_max
        y_in = y > y_min and y < y_max
        
        in_bounds = x_in and y_in
        return in_bounds
    
    def id_as_int(self, id: str) -> np.ndarray:
        id_bytes = id.encode(ENCODING)
        return np.frombuffer(id_bytes, dtype=np.int32)

    def int_as_id(self, id: np.ndarray) -> str:
        id_bytes = np.frombuffer(id, dtype=np.int8).tobytes()
        return id_bytes.decode(ENCODING)
