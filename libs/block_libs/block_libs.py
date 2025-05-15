from typing import Dict, List
import torch

from transformers import AutoModelForCausalLM
from libs.utils.logging_utils import logger
from libs.block_libs.types_and_structs import LayerLevelBlockType


def _mean_abs(grad_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute absolute mean over last two dimensions.
    """
    # print_rank_0(f"use mean()abs() as calculation strategy", global_rank)
    return grad_tensor.mean(dim=(1, 3)).abs()


def calculate_targeted_module_dims(model: AutoModelForCausalLM, block_dimension: int) \
    -> Dict[str, List[int]]:
    targeted_module_dims = {}

    TARGET_MODULE_NAMES = {
        'gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj',
        'o_proj'
    }

    for name, param in model.named_parameters():
        if ("mlp" in name or "attn" in name) and "weight" in name:
            for target_module_name in TARGET_MODULE_NAMES:
                if target_module_name in name and target_module_name not in targeted_module_dims:
                    targeted_module_dims[target_module_name] = [
                        int(param.shape[0] / block_dimension),
                        int(param.shape[1] / block_dimension)
                    ]
                    break

    logger.info(f"targeted_module_dims: {targeted_module_dims}")
    return targeted_module_dims


def get_total_num_blocks(model: AutoModelForCausalLM,
                         block_dimension: int) -> int:
    num_total_blocks = 0
    for _, param in model.named_parameters():
        if isinstance(param, torch.Tensor) and param.ndim == 2:
            num_total_blocks += param.shape[0] / block_dimension * param.shape[
                1] / block_dimension

    return int(num_total_blocks)


def calculate_mean_grad_per_block(warup_abs_grads: LayerLevelBlockType,
                                  targeted_module_dims: Dict[str, List[int]],
                                  block_dimension: int) -> Dict:
    block_means = {}
    for key, grad in warup_abs_grads.items():
        targeted_module_name = key[0]
        targeted_module_dim1 = targeted_module_dims[targeted_module_name][0]
        targeted_module_dim2 = targeted_module_dims[targeted_module_name][1]

        reshaped_grad = grad.reshape(targeted_module_dim1, block_dimension,
                                     targeted_module_dim2, block_dimension)
        block_means[key] = _mean_abs(reshaped_grad)

    return block_means
