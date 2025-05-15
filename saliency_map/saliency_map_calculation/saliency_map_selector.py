from collections import defaultdict
import heapq
import os
from block_libs.block_dimention_calculation import calculate_block_dimension
from block_libs.block_libs import calculate_mean_grad_per_block, calculate_targeted_module_dims, get_total_num_blocks
from smt.smt_calculation.smt_gradient_plotter import plot_gradient_per_block_distribution
from utils.logging_utils import logger
from block_libs.types_and_structs import LayerLevelBlockType, ModuleType, SelectedSubmatrixType
from transformers import AutoModelForCausalLM


def select_submatrix_based_on_grads(
        output_dir: str, model: AutoModelForCausalLM,
        warmup_grads: LayerLevelBlockType, downsample_blocks_ratio: float,
        enable_analysis: float) -> SelectedSubmatrixType:

    block_dimension = calculate_block_dimension(model)
    logger.info(f"block_size is {block_dimension}")

    layer_block_grad_analysis_path = os.path.join(output_dir,
                                                  f'layer_level_grads_plots')
    os.makedirs(layer_block_grad_analysis_path, exist_ok=True)

    targeted_module_dims = calculate_targeted_module_dims(
        model, block_dimension)
    num_total_blocks = get_total_num_blocks(model, block_dimension)

    block_means = calculate_mean_grad_per_block(warmup_grads,
                                                targeted_module_dims,
                                                block_dimension)

    selected_submatrix = select_blocks(downsample_blocks_ratio,
                                       enable_analysis,
                                       layer_block_grad_analysis_path,
                                       num_total_blocks, block_means)

    return selected_submatrix


def select_blocks(downsample_blocks_ratio: float, enable_analysis: bool,
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
