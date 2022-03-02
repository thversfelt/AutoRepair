from typing import Dict
import torch
from benchmark.agent import EgoAgent, VehicleAgent
from benchmark.custom_map_api import CustomMapAPI
from benchmark.lane import Lane


class Scene:
    
    def __init__(self, index: int, id: int, map: CustomMapAPI):
        self.index: int = index
        self.id: int = id
        self.map: CustomMapAPI = map
        self.ego: EgoAgent = None
        self.agents: Dict[VehicleAgent] = {}
        self.world_from_ego = None
        self.ego_from_world = None

    def update(self, data_batch: Dict[str, torch.Tensor]):
        # Update the ego.
        if self.ego is None:
            self.ego = EgoAgent(self.map, self.index)
        self.ego.update(data_batch)
        
        # Update the agents.
        agents_ids = data_batch["all_other_agents_track_ids"][self.index].cpu().numpy()
        for agent_index, agent_id in enumerate(agents_ids):
            # Ensure the agent has a valid id (an id of 0 is invalid).
            if agent_id == 0:
                continue
            
            if agent_id not in self.agents:
                self.agents[agent_id] = VehicleAgent(self.map , self.index, agent_index, agent_id)
            
            agent = self.agents[agent_id]
            agent.update(data_batch)
            
        # Update transformation matrices.
        self.world_from_ego = data_batch["world_from_agent"][self.index].cpu().numpy()
        self.ego_from_world = data_batch["agent_from_world"][self.index].cpu().numpy()
        