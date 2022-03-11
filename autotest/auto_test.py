import os
import torch

from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric)
from l5kit.cle.validators import RangeValidator, ValidationCountingAggregator
from typing import List
from bokeh.io import show
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDatasetVectorized
from l5kit.simulation.unroll import ClosedLoopSimulator
from l5kit.simulation.dataset import SimulationConfig
from l5kit.visualization.visualizer.visualizer import visualize
from l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scene
from autotest.model.evaluation.metrics import CollisionMetric, SafeDistanceMetric
from autotest.model.model import Model
from autotest.util.map_api import CustomMapAPI
from autotest.util.vectorizer import CustomVectorizer
from prettytable import PrettyTable


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
        
        # Assign metrics.
        metrics = [
            CollisionMetric(),
            SafeDistanceMetric()
        ]
        
        # Load the ego model.
        self.model = Model(map, metrics).to(device)

        sim_config = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,
                                distance_th_far=500, distance_th_close=50, num_simulation_steps=248,
                                start_frame_index=0, show_info=True)

        self.sim = ClosedLoopSimulator(sim_config, vectorized_dataset, device, self.model, model_agents=None)
    
    def run(self, scene_ids: List[int], aggregated=True, visualized=False) -> dict:
        
        # Reset the model.
        self.model.reset()
        
        # Unroll the simulation.
        sim_outs = self.sim.unroll(scene_ids)
        results = self.model.evaluation.results
        
        # Visualize the simulation.
        if visualized:
            for sim_out in sim_outs:  # for each scene
                vis_in = simulation_out_to_visualizer_scene(sim_out, self.model.map)
                show(visualize(sim_out.scene_id, vis_in))
        
        # Aggregate the metric scores.
        if aggregated:
            aggregated_results = {}
            for scene_id, _ in results.items():
                aggregated_results[scene_id] = {}
                for metric_name, scores in results[scene_id].items():
                    aggregated_results[scene_id][metric_name] = min(scores)
            return aggregated_results
             
        # Return the resulting metric scores.
        return results