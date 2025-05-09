from typing import DefaultDict, Dict, Tuple, List

import torch

# Example: {"('down_proj', 1)":[[4,21],[0,21]]}
SelectedSubmatrixCoordinatesType = DefaultDict[Tuple[str, int],
                                               List[Tuple[int, int]]]

# Example: {('gate_proj', 0): tensor([[...]]), ...}
LayerLevelGradType = Dict[Tuple[str, int], torch.Tensor]
