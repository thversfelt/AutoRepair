import os
import random

import torch
from bokeh.io import show
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.data import MapAPI
from l5kit.dataset import EgoDataset, EgoDatasetVectorized
from l5kit.rasterization.rasterizer_builder import build_rasterizer
from l5kit.simulation.dataset import SimulationConfig
from l5kit.visualization.visualizer.visualizer import visualize
from l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scene

from custom_closed_loop_simulator import CustomClosedLoopSimulator
from ego_model import EgoModel
from lyftl5.custom_map_api import CustomMapAPI
from lyftl5.custom_vectorizer import CustomVectorizer

if __name__ == '__main__':
    # set env variable for data.
    os.environ["L5KIT_DATA_FOLDER"] = "lyftl5/tmp/l5kit_data"
    dm = LocalDataManager(None)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: " + str(device))
    torch.set_grad_enabled(False)

    # Initialize the agents dataset.
    agents_cfg = load_config_data("lyftl5/agents_config.yaml")
    eval_agents_cfg = agents_cfg["val_data_loader"]
    eval_agents_zarr = ChunkedDataset(dm.require(eval_agents_cfg["key"])).open()
    rasterizer = build_rasterizer(agents_cfg, dm)
    agents_dataset = EgoDataset(agents_cfg, eval_agents_zarr, rasterizer)
    print(agents_dataset)

    # Initialize the ego dataset.
    ego_cfg = load_config_data("lyftl5/ego_config.yaml")
    eval_ego_cfg = ego_cfg["val_data_loader"]
    eval_ego_zarr = ChunkedDataset(dm.require(eval_ego_cfg["key"])).open()
    map_api = CustomMapAPI(ego_cfg, dm)
    vectorizer = CustomVectorizer(ego_cfg, map_api)
    ego_dataset = EgoDatasetVectorized(ego_cfg, eval_ego_zarr, vectorizer)
    print(ego_dataset)

    # Load the pre-trained agents model.
    agents_model_path = "lyftl5/tmp/agents_model/agents_model.pt"
    agents_model = torch.load(agents_model_path).to(device)
    agents_model = agents_model.eval()

    # Load the ego model.
    ego_model = EgoModel(map_api).to(device)
    ego_model = ego_model.eval()

    # Setup the simulation.
    num_scenes_to_unroll = 2
    num_simulation_steps = 248  # 248 is the maximum amount simulation steps.
    scenes_to_unroll = [0, 1] #random.sample(range(0, len(eval_ego_zarr.scenes)), num_scenes_to_unroll)

    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                               distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                               start_frame_index=0, show_info=True)

    sim_loop = CustomClosedLoopSimulator(sim_cfg, device, agents_dataset, ego_dataset, agents_model, ego_model)

    # Unroll.
    sim_outs = sim_loop.unroll(scenes_to_unroll)

    # Visualize.
    semantic_map = MapAPI.from_cfg(dm, ego_cfg)
    for sim_out in sim_outs:  # for each scene
        vis_in = simulation_out_to_visualizer_scene(sim_out, semantic_map)
        show(visualize(sim_out.scene_id, vis_in))
