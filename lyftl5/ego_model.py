import torch
import torch.nn as nn
from typing import Dict

class EgoModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # The ego agent will simply stand still.
        eval_dict = {
            "positions": data_batch['history_positions'], 
            "yaws": data_batch['history_yaws']
        }
        return eval_dict
