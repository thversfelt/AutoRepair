import torch
import torch.nn as nn
from typing import Dict

class EgoAgent(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        #eval_dict = {"positions": [], "yaws": []}
        return data_batch
