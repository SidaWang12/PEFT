from enum import Enum, auto
from typing import DefaultDict, Dict, Tuple, List

import torch

# Example: {"('down_proj', 1)":[[4,21],[0,21]]}
SelectedSubmatrixType = DefaultDict[Tuple[str, int], List[Tuple[int, int]]]

# Example: {('gate_proj', 0): tensor([[...]]), ...}
LayerLevelBlockType = Dict[Tuple[str, int], torch.Tensor]


class ModuleType(Enum):
    MLP = auto()
    ATTENTION = auto()
