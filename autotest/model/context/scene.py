import torch

from typing import Dict
from autotest.model.context.agent import EgoAgent, VehicleAgent
from autotest.util.map_api import CustomMapAPI


class Scene:
    
    def __init__(self, index: int, id: int, map: CustomMapAPI):
        self.index: int = index
        self.id: int = id
        self.map: CustomMapAPI = map
        self.ego: EgoAgent = None
        self.agents: Dict[int, VehicleAgent] = {}

    def update(self, data_batch: Dict[str, torch.Tensor]):
        self.update_agents(data_batch)
        self.update_ego(data_batch)
    
    def update_ego(self, data_batch: Dict[str, torch.Tensor]):
        # Update the ego.
        if self.ego is None:
            self.ego = EgoAgent(self.map, self.index)
        
        self.ego.update(data_batch, self.agents)
    
    def update_agents(self, data_batch: Dict[str, torch.Tensor]):
        # Get the agent ids of the agents that are updated in this frame (data batch).
        updated_agents_ids = data_batch["all_other_agents_track_ids"][self.index].cpu().numpy()
        
        # Update these agents.
        for agent_index, agent_id in enumerate(updated_agents_ids):
            # Ensure the agent has a valid id (an id of 0 is invalid).
            if agent_id == 0:
                continue
            
            # If the agent does not exist in the dictionary of agents, create it and add it.
            if agent_id not in self.agents:
                self.agents[agent_id] = VehicleAgent(self.map , self.index, agent_id)
            
            agent = self.agents[agent_id]
            agent.update(data_batch, agent_index)
            
        # Remove the agents that were in a previous frame, but not in this frame anymore (outdated).
        all_agents_ids = list(self.agents.keys())
        outdated_agents_ids = list(set(all_agents_ids) - set(updated_agents_ids))
        for agent_id in outdated_agents_ids:
            del self.agents[agent_id]
