import os

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
from lyftl5.custom_vectorizer import build_custom_vectorizer

if __name__ == '__main__':
    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = "./tmp/l5kit_data"
    dm = LocalDataManager(None)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: " + str(device))

    num_scenes_to_unroll = 2
    num_simulation_steps = 50

    agents_cfg = load_config_data("agents_config.yaml")
    agents_model_path = "./tmp/agents_model/agents_model.pt"
    agents_model = torch.load(agents_model_path).to(device)
    agents_model = agents_model.eval()

    ego_cfg = load_config_data("ego_config.yaml")
    ego_model = EgoModel().to(device)
    ego_model = ego_model.eval()

    torch.set_grad_enabled(False)

    # ===== INIT DATASET
    eval_agents_cfg = agents_cfg["val_data_loader"]
    eval_agents_zarr = ChunkedDataset(dm.require(eval_agents_cfg["key"])).open()
    rasterizer = build_rasterizer(agents_cfg, dm)
    agents_dataset = EgoDataset(agents_cfg, eval_agents_zarr, rasterizer)
    print(agents_dataset)

    eval_ego_cfg = ego_cfg["val_data_loader"]
    eval_ego_zarr = ChunkedDataset(dm.require(eval_ego_cfg["key"])).open()
    vectorizer = build_custom_vectorizer(ego_cfg, dm)
    ego_dataset = EgoDatasetVectorized(ego_cfg, eval_ego_zarr, vectorizer)
    print(ego_dataset)

    # ==== DEFINE CLOSED-LOOP SIMULATION
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=True,
                               distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                               start_frame_index=0, show_info=True)

    sim_loop = CustomClosedLoopSimulator(sim_cfg, device, agents_dataset, ego_dataset, agents_model, ego_model)

    # ==== UNROLL
    scenes_to_unroll = list(range(0, len(eval_ego_zarr.scenes), len(eval_ego_zarr.scenes) // num_scenes_to_unroll))
    sim_outs = sim_loop.unroll(scenes_to_unroll)

    semantic_map = MapAPI.from_cfg(dm, ego_cfg)
    for sim_out in sim_outs:  # for each scene
        vis_in = simulation_out_to_visualizer_scene(sim_out, semantic_map)
        show(visualize(sim_out.scene_id, vis_in))
