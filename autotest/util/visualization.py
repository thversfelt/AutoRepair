import matplotlib.pyplot as plt
import numpy as np

def visualize_lanes_spatial_index(map_api):
    plt.figure()
    
    for _, lane_id in enumerate(map_api.lanes_ids):
        lane_polygon = map_api.get_lane_polygon(lane_id)
        xs, ys = lane_polygon.exterior.coords.xy
        plt.plot(xs,ys, color='green')

    for leaf in map_api.lanes_spatial_index.leaves():
        x_min = leaf[2][0]
        y_min = leaf[2][1]
        x_max = leaf[2][2]
        y_max = leaf[2][3]
        
        leaf_bounds = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
        xs, ys = zip(*leaf_bounds)
        plt.plot(xs,ys, color='red')
        
    plt.axis('scaled')
    plt.show()
    
def visualize_map_matching(map_api, visited_lanes_ids, candidate_lanes_ids, trajectory):
    plt.figure()
    
    for visited_lane_id in visited_lanes_ids:
        lane_polygon = map_api.get_lane_polygon(visited_lane_id)
        xs, ys = lane_polygon.exterior.coords.xy
        plt.plot(xs,ys, color='green')
    for candidate_lane_id in candidate_lanes_ids:
        lane_polygon = map_api.get_lane_polygon(candidate_lane_id)
        xs, ys = lane_polygon.exterior.coords.xy
        plt.plot(xs,ys, color='red')
    for position in trajectory:
        x = position[0]
        y = position[1]
        plt.scatter(x, y, color='black', s=1)
        
    plt.axis('scaled')
    plt.show()