from collections import defaultdict
from functools import reduce
import math
import os
import heapq
from typing import Dict, List
from matplotlib import pyplot as plt
import torch
from utils.logging import logger
from utils.types_and_structs import LayerLevelGradType, SMTBlockType, SelectedSubmatrixType
from smt_gradient.smt_gradient_plotter import generate_grad_heatmaps, plot_layer_level_grads, plot_gradient_per_block_distribution
from transformers import AutoModelForCausalLM


def select_submatrix(
        model: AutoModelForCausalLM, warmup_grads: LayerLevelGradType,
        global_step: int, enable_analysis: bool, output_dir: str,
        downsample_blocks_ratio: float,
        mlp_or_attention: SMTBlockType) -> SelectedSubmatrixType:
    warup_abs_grads = {}
    for key in warmup_grads:
        warup_abs_grads[key] = warmup_grads[key].abs() / global_step

    if enable_analysis:
        grads_heatmap_path = os.path.join(output_dir, "grad_heatmaps")
        generate_grad_heatmaps(grads_heatmap_path, warup_abs_grads)

    layer_block_grad_analysis_path = os.path.join(output_dir, f'{mlp_or_attention.name}_plots')
    os.makedirs(layer_block_grad_analysis_path, exist_ok=True)

    if enable_analysis:
        _analyze_layer_level_grads(layer_block_grad_analysis_path, warup_abs_grads,
                                   mlp_or_attention)

    block_dimension = _get_gcd_from_weight_shape(model)
    logger.info(f"block_size is {block_dimension}")

    targeted_module_dims = _calculate_targeted_module_dims(
        model, block_dimension)
    num_total_blocks = _get_total_num_blocks(model, block_dimension)

    block_means = _calculate_mean_grad_per_block(warup_abs_grads,
                                                 targeted_module_dims,
                                                 block_dimension)

    selected_submatrix = _select_submatrix_based_on_grads(
        downsample_blocks_ratio, enable_analysis, layer_block_grad_analysis_path,
        num_total_blocks, block_means)

    return selected_submatrix


def _get_gcd_from_weight_shape(model: AutoModelForCausalLM) -> int:
    """
    Computes the greatest common divisor (GCD) of all tensor dimensions
    in MLP and attention layers.
    """
    all_dims = []
    for name, param in model.named_parameters():
        if "mlp" in name or "attn" in name:
            all_dims.extend(param.shape)
    return reduce(math.gcd, all_dims)


def _analyze_layer_level_grads(output_dir: str,
                               warup_abs_grads: LayerLevelGradType,
                               mlp_or_attention: SMTBlockType) -> None:
    """
    Analyze and plot per-layer gradient statistics.
    """
    grad_statistics = {}
    for key in warup_abs_grads:
        grad_statistics[key] = warup_abs_grads[key].mean()
    plot_layer_level_grads(grad_statistics, output_dir, mlp_or_attention,
                           "mean")

    grad_statistics = {}
    for key in warup_abs_grads:
        grad_statistics[key] = warup_abs_grads[key].var(
        ) / warup_abs_grads[key].mean()
    plot_layer_level_grads(grad_statistics, output_dir, mlp_or_attention,
                           "var-divide-by-mean")


def _mean_abs(grad_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute absolute mean over last two dimensions.
    """
    # print_rank_0(f"use mean()abs() as calculation strategy", global_rank)
    return grad_tensor.mean(dim=(1, 3)).abs()


def _calculate_targeted_module_dims(model: AutoModelForCausalLM, block_dimension: int) \
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


def _get_total_num_blocks(model: AutoModelForCausalLM,
                          block_dimension: int) -> int:
    num_total_blocks = 0
    for _, param in model.named_parameters():
        if isinstance(param, torch.Tensor) and param.ndim == 2:
            num_total_blocks += param.shape[0] / block_dimension * param.shape[
                1] / block_dimension

    return int(num_total_blocks)


def _calculate_mean_grad_per_block(warup_abs_grads: LayerLevelGradType,
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


def _select_submatrix_based_on_grads(
        downsample_blocks_ratio: float, enable_analysis: bool,
        layer_block_grad_analysis_path: str, num_total_blocks,
        block_means) -> SelectedSubmatrixType:
    logger.info(f"num_total_blocks {num_total_blocks}")
    num_selected_mlp_blocks = int(num_total_blocks * downsample_blocks_ratio)
    logger.info(f"num_selected_mlp_blocks {num_selected_mlp_blocks}")

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
                if len(top_blocks) < num_selected_mlp_blocks:
                    heapq.heappush(top_blocks, (abs_mean, (key, i, j)))
                else:
                    heapq.heappushpop(top_blocks, (abs_mean, (key, i, j)))

    if enable_analysis:
        plot_gradient_per_block_distribution(layer_block_grad_analysis_path,
                                             gradients_per_block)

    top_blocks.sort(reverse=True)
    selected_ranked_block = defaultdict(list)
    for mean, (info, row, col) in top_blocks:
        selected_ranked_block[info].append((row, col))

    return selected_ranked_block
