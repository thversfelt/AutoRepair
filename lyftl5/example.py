import os
import random
from l5kit.simulation.unroll import ClosedLoopSimulator

import torch
from bokeh.io import show
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.data import MapAPI
from l5kit.dataset import EgoDatasetVectorized
from l5kit.simulation.dataset import SimulationConfig
from l5kit.visualization.visualizer.visualizer import visualize
from l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scene

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

    # Initialize the ego dataset.
    ego_cfg = load_config_data("lyftl5/ego_config.yaml")
    eval_ego_cfg = ego_cfg["val_data_loader"]
    eval_ego_zarr = ChunkedDataset(dm.require(eval_ego_cfg["key"])).open()
    map_api = CustomMapAPI(ego_cfg, dm)
    vectorizer = CustomVectorizer(ego_cfg, map_api)
    ego_dataset = EgoDatasetVectorized(ego_cfg, eval_ego_zarr, vectorizer)
    print(ego_dataset)
    
    # Load the ego model.
    ego_model = EgoModel(map_api).to(device)
    ego_model = ego_model.eval()

    # Setup the simulation.
    num_scenes_to_unroll = 5
    num_simulation_steps = ego_cfg["model_params"]["future_num_frames"]  # Only simulate for the number of future frames.
    scenes_to_unroll = random.sample(range(0, len(eval_ego_zarr.scenes)), num_scenes_to_unroll)
    #scenes_to_unroll = [8]
    print(scenes_to_unroll)

    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,
                               distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                               start_frame_index=0, show_info=True)

    sim_loop = ClosedLoopSimulator(sim_cfg, ego_dataset, device, model_ego=ego_model, model_agents=None)

    # Unroll.
    sim_outs = sim_loop.unroll(scenes_to_unroll)

    # Visualize.
    semantic_map = MapAPI.from_cfg(dm, ego_cfg)
    for sim_out in sim_outs:  # for each scene
        vis_in = simulation_out_to_visualizer_scene(sim_out, semantic_map)
        show(visualize(sim_out.scene_id, vis_in))
