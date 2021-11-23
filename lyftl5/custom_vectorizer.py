from typing import List, Optional

import numpy as np

from l5kit.configs.config import load_metadata
from l5kit.data import DataManager
from l5kit.vectorization.vectorizer import Vectorizer

from lyftl5.custom_map_api import CustomMapAPI


class CustomVectorizer(Vectorizer):
    def _vectorize_map(self, agent_centroid_m: np.ndarray, agent_from_world: np.ndarray,
                       history_tl_faces: List[np.ndarray]) -> dict:
        map_features = super()._vectorize_map(agent_centroid_m, agent_from_world, history_tl_faces)
        additional_map_features = {}

        # TODO: add lane speed limits, connected lane id's, lane_change_to_left/right_id, and lane_yield_to lane id's
        #  from the mapAPI lane elements.

        return {**map_features, **additional_map_features}


def build_custom_vectorizer(cfg: dict, data_manager: DataManager) -> CustomVectorizer:
    """Same as build_vectorizer from l5kit, but then using my custom vectorizer."""
    dataset_meta_key = cfg["raster_params"]["dataset_meta_key"]  # TODO positioning of key
    dataset_meta = load_metadata(data_manager.require(dataset_meta_key))
    world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

    custom_map_api = CustomMapAPI(data_manager.require(cfg["raster_params"]["semantic_map_key"]), world_to_ecef)
    custom_vectorizer = CustomVectorizer(cfg, custom_map_api)

    return custom_vectorizer
