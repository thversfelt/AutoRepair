from typing import Dict
import torch
from l5kit.data import MapAPI
from torch import nn


class EgoModelLaneKeeping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        num_of_scenes = len(data_batch['scene_index'])
        steer = torch.zeros([num_of_scenes], dtype=torch.float64)
        for scene_idx in range(num_of_scenes):
            current_lane_idx = self.get_current_lane_idx(scene_idx, data_batch)
            current_lane_next_mid_point_idx = self.get_next_mid_point_idx(scene_idx, current_lane_idx, data_batch)
            current_lane_next_mid_point = data_batch['lanes_mid'][scene_idx][current_lane_idx][current_lane_next_mid_point_idx]
            # print("current_lane_idx: " + str(current_lane_idx))
            # print("current_lane_next_mid_point_idx: " + str(current_lane_next_mid_point_idx))
            steer[scene_idx] = current_lane_next_mid_point[1]  # Proportionate to Y-coordinate.
        eval_dict = {"steer": steer}
        return eval_dict

    def get_current_lane_idx(self, scene_idx, data_batch: Dict[str, torch.Tensor]):
        current_lane_idx = None
        current_lane_next_mid_point_distance = None
        for lane_idx, _ in enumerate(data_batch['lanes_mid'][scene_idx]):
            lane_next_mid_point_idx = self.get_next_mid_point_idx(scene_idx, lane_idx, data_batch)
            if lane_next_mid_point_idx is None:
                continue  # Could not find a next mid point for this lane, continue to look at other lanes.
            lane_next_mid_point = data_batch['lanes_mid'][scene_idx][lane_idx][lane_next_mid_point_idx]
            lane_next_mid_point_distance = torch.linalg.norm(lane_next_mid_point)
            if current_lane_idx is None:
                current_lane_idx = lane_idx
                current_lane_next_mid_point_distance = lane_next_mid_point_distance
            if current_lane_next_mid_point_distance < current_lane_next_mid_point_distance:
                current_lane_idx = lane_idx
                current_lane_next_mid_point_distance = lane_next_mid_point_distance
        return current_lane_idx

    def get_next_mid_point_idx(self, scene_idx: int, lane_idx: int, data_batch: Dict[str, torch.Tensor]):
        next_mid_point_idx = None
        next_mid_point_distance = None
        for mid_point_idx, _ in enumerate(data_batch['lanes_mid'][scene_idx][lane_idx]):
            mid_point = data_batch['lanes_mid'][scene_idx][lane_idx][mid_point_idx]
            if mid_point[0] <= 0:  # X-coordinate is equal or less than 0 in the agent's reference system, so is behind.
                continue  # Filter points that are behind the agent in its reference system.
            mid_point_distance = torch.linalg.norm(mid_point)
            if next_mid_point_idx is None:
                next_mid_point_idx = mid_point_idx
                next_mid_point_distance = mid_point_distance
            elif mid_point_distance < next_mid_point_distance:
                next_mid_point_idx = mid_point_idx
                next_mid_point_distance = mid_point_distance
        return next_mid_point_idx
