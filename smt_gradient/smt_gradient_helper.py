from collections import defaultdict
import re
import heapq
from typing import DefaultDict, Dict, List, Tuple
from deepspeed.utils import safe_get_full_grad
import torch
from helpers.logging import logger
from smt_gradient.smt_gradient_plotter import *


def get_warmup_grads(
        model: torch.nn.Module, warmup_grads: Dict[Tuple[str, int],
                                                   torch.Tensor]) -> None:
    """
    Accumulate warmup gradients for MLP layers across steps.
    """
    pattern = re.compile(r'model\.layers\.(\d+)\.')

    for name, param in model.named_parameters():
        match = pattern.search(name)
        layer_number = int(match.group(1)) if match else None
        if 'mlp' in name and 'weight' in name:
            grad = safe_get_full_grad(param)  # (hidden_dim, head_dim)
            module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
            key = (module_name, layer_number)

            #defaultdict(torch.float32)
            if key not in warmup_grads:
                # warmup_grads[(module_name, layer_number)] = grad.detach().to(torch.float32)
                warmup_grads[key] = grad.detach().cpu().to(torch.float32)

            else:
                warmup_grads[key] += grad.detach().cpu().to(torch.float32)
                # warmup_grads[(module_name, layer_number)] += grad.detach().to(torch.float32)
                # del grad


def analyze_layer_level_grads(
        output_dir: str, warmup_grads: Dict[Tuple[str, int],
                                            torch.Tensor]) -> None:
    """
    Analyze and plot per-layer gradient statistics.
    """
    grad_statistics = {}
    for key in warmup_grads:
        grad_statistics[key] = warmup_grads[key].mean()
    plot_layer_level_grads(grad_statistics, output_dir, "mean")

    grad_statistics = {}
    for key in warmup_grads:
        grad_statistics[key] = warmup_grads[key].var(
        ) / warmup_grads[key].mean()
    plot_layer_level_grads(grad_statistics, output_dir, "var-divide-by-mean")


def _mean_abs(grad_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute absolute mean over last two dimensions.
    """
    # print_rank_0(f"use mean()abs() as calculation strategy", global_rank)
    return grad_tensor.mean(dim=(1, 3)).abs()


def select_submatrix_based_on_grads(
    model: torch.nn.Module, warup_abs_grads: Dict[Tuple[str, int],
                                                  torch.Tensor],
    block_dimension: int, downsample_attention_blocks_ratio: float,
    enable_analysis: bool, analysis_plot_path: str
) -> DefaultDict[Tuple[str, int], List[Tuple[int, int]]]:
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

    num_total_blocks = 0
    for name, param in model.named_parameters():
        if isinstance(param, torch.Tensor) and param.ndim == 2:
            num_total_blocks += param.shape[0] / block_dimension * param.shape[
                1] / block_dimension

    logger.info(f"num_total_blocks {num_total_blocks}")
    num_selected_blocks = int(num_total_blocks *
                              downsample_attention_blocks_ratio)
    logger.info(f"num_selected_blocks {num_selected_blocks}")

    block_means = {}
    for key, grad in warup_abs_grads.items():
        targeted_module_name = key[0]
        targeted_module_dim1 = targeted_module_dims[targeted_module_name][0]
        targeted_module_dim2 = targeted_module_dims[targeted_module_name][1]

        reshaped_grad = grad.reshape(targeted_module_dim1, block_dimension,
                                     targeted_module_dim2, block_dimension)
        block_means[key] = _mean_abs(reshaped_grad)

    # Use a min-heap to maintain top n blocks efficiently
    top_blocks = []

    gradients_per_block = {}
    for key in block_means.keys():
        gradients_per_block[key[0]] = []

    for key, block_mean in block_means.items():
        for i in range(block_mean.shape[0]):
            for j in range(block_mean.shape[1]):
                abs_mean = block_mean[i, j].item()
                gradients_per_block[key[0]].append(abs_mean)
                if len(top_blocks) < num_selected_blocks:
                    heapq.heappush(top_blocks, (abs_mean, (key, i, j)))
                else:
                    heapq.heappushpop(top_blocks, (abs_mean, (key, i, j)))

    if enable_analysis:
        plot_gradient_per_block_distribution(analysis_plot_path,
                                             gradients_per_block)

    top_blocks.sort(reverse=True)
    selected_ranked_block = defaultdict(list)
    for mean, (info, row, col) in top_blocks:
        selected_ranked_block[info].append((row, col))

    return selected_ranked_block
