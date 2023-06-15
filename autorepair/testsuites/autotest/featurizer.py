from typing import Dict
from autotest.scene import Scene

import numpy as np
import l5kit.data.proto.road_network_pb2 as road_network
import pickle


def featurize_scenes(scenes: Dict[int, Scene]):
    scenes_features = {}
    features_names = [
        "Number of Agents",
        "Direct Distance",
        "Road Distance",
        "Number of Left Turns",
        "Number of Right Turns",
        "Number of Straight Segments",
        "Total Angle",
        "Median Angle",
        "Std Angle",
        "Max Angle",
        "Min Angle",
        "Mean Angle"
    ]

    # Calculate the features values for each scene.
    for scene_id, scene in scenes.items():
        number_of_frames = len(scene.ego.replay_positions)

        # Initialize the scene features dictionary.
        scenes_features[scene_id] = {}
        scenes_features[scene_id]["Number of Agents"] = len(scene.agents)
        scenes_features[scene_id]["Direct Distance"] = np.linalg.norm(scene.ego.replay_positions[0] - scene.ego.replay_positions[number_of_frames - 1])
        scenes_features[scene_id]["Road Distance"] = 0
        scenes_features[scene_id]["Number of Straight Segments"] = 0
        scenes_features[scene_id]["Number of Left Turns"] = 0
        scenes_features[scene_id]["Number of Right Turns"] = 0
        scenes_features[scene_id]["Total Angle"] = 0
        scenes_features[scene_id]["Median Angle"] = 0
        scenes_features[scene_id]["Std Angle"] = 0
        scenes_features[scene_id]["Max Angle"] = 0
        scenes_features[scene_id]["Min Angle"] = 0
        scenes_features[scene_id]["Mean Angle"] = 0

        if scene.ego.replay_route is not None:
            
            lane_angles = []
            for lane_id in scene.ego.replay_route:
                lane = scene.map.lanes[lane_id]
                lane_angles.append(lane.angle() * 180 / np.pi)

                scenes_features[scene_id]["Road Distance"] += lane.length()
                if lane.turn_type == road_network.Lane.TurnType.THROUGH:
                    scenes_features[scene_id]["Number of Straight Segments"] += 1
                elif lane.turn_type == road_network.Lane.TurnType.LEFT or lane.turn_type == road_network.Lane.TurnType.SHARP_LEFT:
                    scenes_features[scene_id]["Number of Left Turns"] += 1
                elif lane.turn_type == road_network.Lane.TurnType.RIGHT or lane.turn_type == road_network.Lane.TurnType.SHARP_RIGHT:
                    scenes_features[scene_id]["Number of Right Turns"] += 1
        
            scenes_features[scene_id]["Total Angle"] = np.sum(lane_angles)
            scenes_features[scene_id]["Median Angle"] = np.median(lane_angles)
            scenes_features[scene_id]["Std Angle"] = np.std(lane_angles)
            scenes_features[scene_id]["Max Angle"] = np.max(lane_angles)
            scenes_features[scene_id]["Min Angle"] = np.min(lane_angles)
            scenes_features[scene_id]["Mean Angle"] = np.mean(lane_angles)
        
    # Replace the features vectors with their corresponding z-scores.
    for feature_name in features_names:
        feature_values = np.array([scenes_features[scene_id][feature_name] for scene_id in scenes.keys()])
        feature_mean =  np.mean(feature_values)
        feature_std = np.std(feature_values)
        
        if feature_std == 0:
            feature_std = 1
        
        for scene_id in scenes.keys():
            scenes_features[scene_id][feature_name] = (scenes_features[scene_id][feature_name] - feature_mean) / feature_std
    
    return scenes_features


def save_scenes_features(scenes_features: dict, dataset_name: str):
    features_path = f"autotest/data/{dataset_name}_features.pkl"
    with open(features_path, 'wb') as file:
        pickle.dump(scenes_features, file)

def load_scenes_features(dataset_name: str):
    features_path = f"autotest/data/{dataset_name}_features.pkl"
    with open(features_path, 'rb') as file:
        scenes_features = pickle.load(file)
    return scenes_features
