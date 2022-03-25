import ast
import os
import torch

from typing import List
from bokeh.io import show
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDatasetVectorized
from l5kit.simulation.unroll import ClosedLoopSimulator
from l5kit.simulation.dataset import SimulationConfig
from l5kit.visualization.visualizer.visualizer import visualize
from l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scene
from autotest.model.evaluation.metrics import CollisionMetric, SafeDistanceMetric, TrafficLightsMetric
from autotest.model.model import Model
from autotest.util.map_api import CustomMapAPI
from autotest.util.vectorizer import CustomVectorizer


class AutoTest:
    def __init__(self) -> None:
        os.environ["L5KIT_DATA_FOLDER"] = "autotest/data"
        dm = LocalDataManager(None)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        print("device: " + str(device))

        # Initialize the dataset.
        config = load_config_data("autotest/model/config.yaml")
        dataset_path = config["val_data_loader"]["key"]
        dataset = ChunkedDataset(dm.require(dataset_path)).open()
        map = CustomMapAPI(config, dm)
        vectorizer = CustomVectorizer(config, map)
        vectorized_dataset = EgoDatasetVectorized(config, dataset, vectorizer)
        print(vectorized_dataset)

        # Load the ego model.
        self.model = Model(map).to(device)

        sim_config = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,
                                distance_th_far=500, distance_th_close=50, num_simulation_steps=248,
                                start_frame_index=0, show_info=True)

        self.sim = ClosedLoopSimulator(sim_config, vectorized_dataset, device, self.model, model_agents=None)
    
    def run(self, rule_set: ast.Module, scene_ids: List[int], visualized=False) -> tuple:
        # Assign metrics.
        metrics = [
            CollisionMetric(),
            SafeDistanceMetric(),
            TrafficLightsMetric()
        ]
        
        # Initialize the model.
        self.model.initialize(rule_set, metrics)
        
        # Unroll the simulation.
        sim_outs = self.sim.unroll(scene_ids)
        
        # Visualize the simulation.
        if visualized:
            for sim_out in sim_outs:  # for each scene
                vis_in = simulation_out_to_visualizer_scene(sim_out, self.model.map)
                show(visualize(sim_out.scene_id, vis_in))
        
        # Obtain the metrics scores, and executed statements per frame per scene.
        metrics_scores = self.model.evaluation.metrics_scores
        executed_statements = self.model.instrumentation.executed_statements

        # Aggregate the results.
        results = {}
        
        for scene_id in scene_ids:
            if scene_id not in results:
                results[scene_id] = {
                    "metrics_scores": [],
                    "executed_statements": []
                }

            num_of_frames = len(executed_statements[scene_id])
            violation_frame_index = num_of_frames - 1
            
            for frame_index in range(num_of_frames):
                if frame_index > violation_frame_index:
                    continue
            
                scores = []
                
                for metric_name in metrics_scores[scene_id].keys():
                    score = metrics_scores[scene_id][metric_name][frame_index]
                    scores.append(score)
                    
                    if score < 0.0:
                        violation_frame_index = frame_index
                
                statements = executed_statements[scene_id][frame_index]
                
                results[scene_id]["metrics_scores"] = scores
                results[scene_id]["executed_statements"] = statements
        
        return results