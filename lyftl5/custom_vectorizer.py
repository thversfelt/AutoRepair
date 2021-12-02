from typing import List

import numpy as np
from l5kit.vectorization.vectorizer import Vectorizer


class CustomVectorizer(Vectorizer):
    def _vectorize_map(self, agent_centroid_m: np.ndarray, agent_from_world: np.ndarray,
                       history_tl_faces: List[np.ndarray]) -> dict:
        """Override the default map vectorizer to return an empty dictionary, so it doesn't perform operations such
        as finding the ego's nearest lane each frame. So, the vectorizer will only vectorize the agents."""
        return {}
