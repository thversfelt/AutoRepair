import numpy as np

from typing import List, Optional
from l5kit.data import filter_agents_by_distance, filter_agents_by_labels, filter_tl_faces_by_status
from l5kit.data.filter import filter_agents_by_track_id, get_other_agents_ids
from l5kit.data.map_api import MapAPI
from l5kit.vectorization.vectorizer import Vectorizer
from l5kit.sampling.agent_sampling import get_relative_poses


class CustomVectorizer(Vectorizer):
    def __init__(self, cfg: dict, mapAPI: MapAPI):
        """Instantiates the class.

        Arguments:
            cfg: config to load settings from
            mapAPI: mapAPI to query map information
        """
        self.mapAPI = mapAPI
        
        self.history_num_frames_agents = cfg["model_params"]["history_num_frames_agents"]
        self.future_num_frames = cfg["model_params"]["future_num_frames"]
        self.history_num_frames_max = max(cfg["model_params"]["history_num_frames_ego"], self.history_num_frames_agents)
        
        self.max_agents_distance = cfg["data_generation_params"]["max_agents_distance"]
        self.other_agents_num = cfg["data_generation_params"]["other_agents_num"]
        
        self.traffic_light_faces_num = cfg["data_generation_params"]["traffic_light_faces_num"]
    
    def _vectorize_map(self, agent_centroid_m: np.ndarray, agent_from_world: np.ndarray,
                       history_tl_faces: List[np.ndarray]) -> dict:
        
        # Filter the list of traffic light faces for active traffic light faces.
        active_traffic_light_faces_ids = filter_tl_faces_by_status(history_tl_faces[0], "ACTIVE")["face_id"]
        
        # Remove duplicate id's.
        active_traffic_light_faces_ids = list(set(active_traffic_light_faces_ids))

        traffic_light_faces_ids = np.zeros(self.traffic_light_faces_num, dtype=np.int32)
        
        for i, active_traffic_light_face_id in enumerate(active_traffic_light_faces_ids):
            traffic_light_faces_ids[i] = self.mapAPI.id_as_int(active_traffic_light_face_id)

        return {
            "traffic_light_faces_ids": traffic_light_faces_ids
        }
        
    def _vectorize_agents(self, selected_track_id: Optional[int], agent_centroid_m: np.ndarray,
                          agent_yaw_rad: float, agent_from_world: np.ndarray, history_frames: np.ndarray,
                          history_agents: List[np.ndarray], history_position_m: np.ndarray,
                          history_yaws_rad: np.ndarray, history_availability: np.ndarray, future_frames: np.ndarray,
                          future_agents: List[np.ndarray]) -> dict:
        """Vectorize agents in a frame.

        Arguments:
            selected_track_id: selected_track_id: Either None for AV, or the ID of an agent that you want to
            predict the future of.
            This agent is centered in the representation and the returned targets are derived from their future states.
            agent_centroid_m: position of the target agent
            agent_yaw_rad: yaw angle of the target agent
            agent_from_world: inverted agent pose as 3x3 matrix
            history_frames: historical frames of the target frame
            history_agents: agents appearing in history_frames
            history_tl_faces: traffic light faces in history frames
            history_position_m: historical positions of target agent
            history_yaws_rad: historical yaws of target agent
            history_availability: availability mask of history frames
            future_frames: future frames of the target frame
            future_agents: agents in future_frames

        Returns:
            dict: a dict containing the vectorized agent representation of the target frame
        """
        # compute agent features
        # sequence_length x 2 (two being x, y)
        agent_points = history_position_m.copy()
        # sequence_length x 1
        agent_yaws = history_yaws_rad.copy()
        # sequence_length x xy+yaw (3)
        agent_trajectory_polyline = np.concatenate([agent_points, agent_yaws], axis=-1)
        agent_polyline_availability = history_availability.copy()

        # get agents around AoI sorted by distance in a given radius. Give priority to agents in the current time step
        history_agents_flat = filter_agents_by_labels(np.concatenate(history_agents))
        cur_agents = filter_agents_by_labels(history_agents[0])

        list_agents_to_take = get_other_agents_ids(
            history_agents_flat["track_id"], cur_agents["track_id"], selected_track_id, self.other_agents_num
        )

        # Loop to grab history and future for all other agents
        all_other_agents_history_positions = np.zeros(
            (self.other_agents_num, self.history_num_frames_max + 1, 2), dtype=np.float32)
        all_other_agents_history_yaws = np.zeros(
            (self.other_agents_num, self.history_num_frames_max + 1, 1), dtype=np.float32)
        all_other_agents_history_extents = np.zeros(
            (self.other_agents_num, self.history_num_frames_max + 1, 2), dtype=np.float32)
        all_other_agents_history_availability = np.zeros(
            (self.other_agents_num, self.history_num_frames_max + 1), dtype=np.float32)
        all_other_agents_types = np.zeros((self.other_agents_num,), dtype=np.int64)
        all_other_agents_track_ids = np.zeros((self.other_agents_num,), dtype=np.int64)

        all_other_agents_future_positions = np.zeros(
            (self.other_agents_num, self.future_num_frames, 2), dtype=np.float32)
        all_other_agents_future_yaws = np.zeros((self.other_agents_num, self.future_num_frames, 1), dtype=np.float32)
        all_other_agents_future_extents = np.zeros((self.other_agents_num, self.future_num_frames, 2), dtype=np.float32)
        all_other_agents_future_availability = np.zeros(
            (self.other_agents_num, self.future_num_frames), dtype=np.float32)

        for idx, track_id in enumerate(list_agents_to_take):
            (
                agent_history_coords_offset,
                agent_history_yaws_offset,
                agent_history_extent,
                agent_history_availability,
            ) = get_relative_poses(self.history_num_frames_max + 1, history_frames, track_id, history_agents,
                                   agent_from_world, agent_yaw_rad)

            all_other_agents_history_positions[idx] = agent_history_coords_offset
            all_other_agents_history_yaws[idx] = agent_history_yaws_offset
            all_other_agents_history_extents[idx] = agent_history_extent
            all_other_agents_history_availability[idx] = agent_history_availability
            # NOTE (@lberg): assumption is that an agent doesn't change class (seems reasonable)
            # We look from history backward and choose the most recent time the track_id was available.
            current_other_actor = filter_agents_by_track_id(history_agents_flat, track_id)[0]
            all_other_agents_types[idx] = np.argmax(current_other_actor["label_probabilities"])
            all_other_agents_track_ids[idx] = track_id

            (
                agent_future_coords_offset,
                agent_future_yaws_offset,
                agent_future_extent,
                agent_future_availability,
            ) = get_relative_poses(
                self.future_num_frames, future_frames, track_id, future_agents, agent_from_world, agent_yaw_rad
            )
            all_other_agents_future_positions[idx] = agent_future_coords_offset
            all_other_agents_future_yaws[idx] = agent_future_yaws_offset
            all_other_agents_future_extents[idx] = agent_future_extent
            all_other_agents_future_availability[idx] = agent_future_availability

        # crop similar to ego above
        all_other_agents_history_positions[:, self.history_num_frames_agents + 1:] *= 0
        all_other_agents_history_yaws[:, self.history_num_frames_agents + 1:] *= 0
        all_other_agents_history_extents[:, self.history_num_frames_agents + 1:] *= 0
        all_other_agents_history_availability[:, self.history_num_frames_agents + 1:] *= 0

        # compute other agents features
        # num_other_agents (M) x sequence_length x 2 (two being x, y)
        agents_points = all_other_agents_history_positions.copy()
        # num_other_agents (M) x sequence_length x 1
        agents_yaws = all_other_agents_history_yaws.copy()
        # agents_extents = all_other_agents_history_extents[:, :-1]
        # num_other_agents (M) x sequence_length x self._vector_length
        other_agents_polyline = np.concatenate([agents_points, agents_yaws], axis=-1)
        other_agents_polyline_availability = all_other_agents_history_availability.copy()

        agent_dict = {
            "all_other_agents_history_positions": all_other_agents_history_positions,
            "all_other_agents_history_yaws": all_other_agents_history_yaws,
            "all_other_agents_history_extents": all_other_agents_history_extents,
            "all_other_agents_history_availability": all_other_agents_history_availability.astype(np.bool),
            "all_other_agents_future_positions": all_other_agents_future_positions,
            "all_other_agents_future_yaws": all_other_agents_future_yaws,
            "all_other_agents_future_extents": all_other_agents_future_extents,
            "all_other_agents_future_availability": all_other_agents_future_availability.astype(np.bool),
            "all_other_agents_types": all_other_agents_types,
            "all_other_agents_track_ids": all_other_agents_track_ids,
            "agent_trajectory_polyline": agent_trajectory_polyline,
            "agent_polyline_availability": agent_polyline_availability.astype(np.bool),
            "other_agents_polyline": other_agents_polyline,
            "other_agents_polyline_availability": other_agents_polyline_availability.astype(np.bool),
        }

        return agent_dict
