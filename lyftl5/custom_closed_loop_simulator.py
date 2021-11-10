from collections import defaultdict
from enum import IntEnum
from typing import DefaultDict, Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm

from l5kit.data import AGENT_DTYPE, PERCEPTION_LABEL_TO_INDEX
from l5kit.dataset import EgoDataset
from l5kit.dataset.utils import move_to_device, move_to_numpy
from l5kit.geometry import rotation33_as_yaw, transform_points
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.simulation.unroll import  ClosedLoopSimulator, ClosedLoopSimulatorModes, SimulationOutput, UnrollInputOutput


class CustomClosedLoopSimulator(ClosedLoopSimulator):
    def __init__(self, 
                 sim_cfg: SimulationConfig,
                 agents_dataset: EgoDataset,
                 ego_dataset: EgoDataset,
                 device: torch.device,
                 model_ego: Optional[torch.nn.Module] = None,
                 model_agents: Optional[torch.nn.Module] = None,
                 keys_to_exclude: Tuple[str] = ("image",),
                 mode: int = ClosedLoopSimulatorModes.L5KIT):
        """
        Create a simulation loop object capable of unrolling ego and agents
        :param sim_cfg: configuration for unroll
        :param agents_dataset: EgoDataset used while unrolling
        :param ego_dataset: EgoDataset used while unrolling
        :param device: a torch device. Inference will be performed here
        :param model_ego: the model to be used for ego
        :param model_agents: the model to be used for agents
        :param keys_to_exclude: keys to exclude from input/output (e.g. huge blobs)
        :param mode: the framework that uses the closed loop simulator
        """
        self.sim_cfg = sim_cfg
        if not sim_cfg.use_ego_gt and model_ego is None and mode == ClosedLoopSimulatorModes.L5KIT:
            raise ValueError("ego model should not be None when simulating ego")
        if not sim_cfg.use_agents_gt and model_agents is None and mode == ClosedLoopSimulatorModes.L5KIT:
            raise ValueError("agents model should not be None when simulating agent")
        if sim_cfg.use_ego_gt and mode == ClosedLoopSimulatorModes.GYM:
            raise ValueError("ego has to be simulated when using gym environment")
        if not sim_cfg.use_agents_gt and mode == ClosedLoopSimulatorModes.GYM:
            raise ValueError("agents need be log-replayed when using gym environment")

        self.model_ego = torch.nn.Sequential().to(device) if model_ego is None else model_ego.to(device)
        self.model_agents = torch.nn.Sequential().to(device) if model_agents is None else model_agents.to(device)

        self.device = device
        self.agents_dataset = agents_dataset
        self.ego_dataset = ego_dataset

        self.keys_to_exclude = set(keys_to_exclude)

    def unroll(self, scene_indices: List[int]) -> List[SimulationOutput]:
        """
        Simulate the dataset for the given scene indices
        :param scene_indices: the scene indices we want to simulate
        :return: the simulated dataset
        """
        sim_agents_dataset = SimulationDataset.from_dataset_indices(self.agents_dataset, scene_indices, self.sim_cfg)
        sim_ego_dataset = SimulationDataset.from_dataset_indices(self.ego_dataset, scene_indices, self.sim_cfg)

        agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]] = defaultdict(list)
        ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]] = defaultdict(list)

        for frame_index in tqdm(range(len(sim_ego_dataset)), disable=not self.sim_cfg.show_info):
            next_frame_index = frame_index + 1
            should_update = next_frame_index != len(sim_ego_dataset)

            # AGENTS
            if not self.sim_cfg.use_agents_gt:
                agents_input = sim_agents_dataset.rasterise_agents_frame_batch(frame_index)
                if len(agents_input):  # agents may not be available
                    agents_input_dict = default_collate(list(agents_input.values()))
                    agents_output_dict = self.model_agents(move_to_device(agents_input_dict, self.device))

                    # for update we need everything as numpy
                    agents_input_dict = move_to_numpy(agents_input_dict)
                    agents_output_dict = move_to_numpy(agents_output_dict)

                    if should_update:
                        self.update_agents(sim_agents_dataset, next_frame_index, agents_input_dict, agents_output_dict)
                        self.update_agents(sim_ego_dataset, next_frame_index, agents_input_dict, agents_output_dict)

                    # update input and output buffers
                    agents_frame_in_out = self.get_agents_in_out(agents_input_dict, agents_output_dict,
                                                                 self.keys_to_exclude)
                    for scene_idx in scene_indices:
                        agents_ins_outs[scene_idx].append(agents_frame_in_out.get(scene_idx, []))

            # EGO
            if not self.sim_cfg.use_ego_gt:
                ego_input = sim_ego_dataset.rasterise_frame_batch(frame_index)
                ego_input_dict = default_collate(ego_input)
                ego_output_dict = self.model_ego(move_to_device(ego_input_dict, self.device))

                ego_input_dict = move_to_numpy(ego_input_dict)
                ego_output_dict = move_to_numpy(ego_output_dict)

                if should_update:
                    self.update_ego(sim_agents_dataset, next_frame_index, ego_input_dict, ego_output_dict)
                    self.update_ego(sim_ego_dataset, next_frame_index, ego_input_dict, ego_output_dict)

                ego_frame_in_out = self.get_ego_in_out(ego_input_dict, ego_output_dict, self.keys_to_exclude)
                for scene_idx in scene_indices:
                    ego_ins_outs[scene_idx].append(ego_frame_in_out[scene_idx])

        simulated_outputs: List[SimulationOutput] = []
        for scene_idx in scene_indices:
            simulated_outputs.append(SimulationOutput(scene_idx, sim_ego_dataset, ego_ins_outs, agents_ins_outs))

        return simulated_outputs