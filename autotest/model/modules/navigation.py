from l5kit.geometry.transform import transform_points
from autotest.model.context.scene import Scene


class Navigation():

    def process(self, scene: Scene) -> float:
        target_point = None
        
        # Follow the ego's original lateral movement as closely as possible.
        trajectory = transform_points(scene.ego.trajectory, scene.ego.ego_from_world)
        trajectory = trajectory[trajectory[:,0] > 0]
        
        if len(trajectory) > 0:
            target_point = trajectory[0]
        else:
            # The ego has moved beyond the ego's original lateral movement, follow the current lane's midpoints.
            for lane_id in scene.ego.route:
                # Get the closest lane midpoints for the ego's current lane.
                lane_midpoints = scene.map.get_closest_lane_midpoints(scene.ego.position, lane_id)
                lane_midpoints = transform_points(lane_midpoints, scene.ego.ego_from_world)
                lane_midpoints = lane_midpoints[lane_midpoints[:,0] > 0]
                
                if len(lane_midpoints) > 0:
                    target_point = lane_midpoints[0]
                    break

        if target_point is not None:
            # Steer input is proportional to the y-coordinate of the closest midpoint.
            steer = 0.5 * target_point[1]
        else:
            steer = 0.0
        
        return steer
