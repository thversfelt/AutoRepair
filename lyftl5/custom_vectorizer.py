from typing import Dict, List, Optional

import numpy as np

from l5kit.data import filter_agents_by_distance, filter_agents_by_labels, filter_tl_faces_by_status
from l5kit.data.filter import filter_agents_by_track_id, get_other_agents_ids
from l5kit.data.map_api import InterpolationMethod, MapAPI
from l5kit.geometry.transform import transform_points
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds
from l5kit.sampling.agent_sampling import get_relative_poses
from l5kit.vectorization.vectorizer import Vectorizer


class CustomVectorizer(Vectorizer):
    def _vectorize_map(self, agent_centroid_m: np.ndarray, agent_from_world: np.ndarray,
                       history_tl_faces: List[np.ndarray]) -> dict:
        """Override the default map vectorizer to return an empty dictionary, so it doesn't perform operations such
        as finding the ego's nearest lane each frame. So, the vectorizer will only vectorize the agents."""
        return {}

    def _vectorize_agents(self, selected_track_id: Optional[int], agent_centroid_m: np.ndarray,
                          agent_yaw_rad: float, agent_from_world: np.ndarray, history_frames: np.ndarray,
                          history_agents: List[np.ndarray], history_position_m: np.ndarray,
                          history_yaws_rad: np.ndarray, history_availability: np.ndarray, future_frames: np.ndarray,
                          future_agents: List[np.ndarray]) -> dict:
        agent_features = super()._vectorize_agents(selected_track_id, agent_centroid_m, agent_yaw_rad, agent_from_world,
                                                history_frames, history_agents, history_position_m, history_yaws_rad,
                                                history_availability, future_frames, future_agents)        
        history_agents_flat = filter_agents_by_labels(np.concatenate(history_agents))
        history_agents_flat = filter_agents_by_distance(history_agents_flat, agent_centroid_m, self.max_agents_distance)

        cur_agents = filter_agents_by_labels(history_agents[0])
        cur_agents = filter_agents_by_distance(cur_agents, agent_centroid_m, self.max_agents_distance)

        list_agents_to_take = get_other_agents_ids(
            history_agents_flat["track_id"], cur_agents["track_id"], selected_track_id, self.other_agents_num
        )
        
        all_other_agents_track_ids = np.zeros(self.other_agents_num, dtype=np.int64)
        all_other_agents_track_ids[:len(list_agents_to_take)] = list_agents_to_take
        additional_agent_features = {"all_other_agents_track_ids": all_other_agents_track_ids}
        
        return {**additional_agent_features, **agent_features}