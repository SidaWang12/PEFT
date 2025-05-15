from collections import defaultdict
import os
import heapq
from libs.block_libs.block_libs import calculate_mean_grad_per_block, calculate_targeted_module_dims, get_total_num_blocks
from libs.block_libs.block_dimention_calculation import calculate_block_dimension
from libs.utils.logging_utils import logger
from libs.block_libs.types_and_structs import LayerLevelBlockType, ModuleType, SelectedSubmatrixType
from libs.plotters.gradient_plotter import generate_grad_heatmaps, plot_layer_level_grads, plot_gradient_per_block_distribution
from transformers import AutoModelForCausalLM


def select_submatrix(model: AutoModelForCausalLM,
                     warmup_grads: LayerLevelBlockType, global_step: int,
                     enable_analysis: bool, output_dir: str,
                     downsample_blocks_ratio: float,
                     mlp_or_attention: ModuleType) -> SelectedSubmatrixType:
    warup_abs_grads = {}
    for key in warmup_grads:
        warup_abs_grads[key] = warmup_grads[key].abs() / global_step

    if enable_analysis:
        grads_heatmap_path = os.path.join(output_dir, "grad_heatmaps")
        generate_grad_heatmaps(grads_heatmap_path, warup_abs_grads)

    layer_block_grad_analysis_path = os.path.join(
        output_dir, f'{mlp_or_attention.name}_plots')
    os.makedirs(layer_block_grad_analysis_path, exist_ok=True)

    if enable_analysis:
        _analyze_layer_level_grads(layer_block_grad_analysis_path,
                                   warup_abs_grads, mlp_or_attention)

    block_dimension = calculate_block_dimension(model)
    logger.info(f"block_size is {block_dimension}")

    targeted_module_dims = calculate_targeted_module_dims(
        model, block_dimension)
    num_total_blocks = get_total_num_blocks(model, block_dimension)

    block_means = calculate_mean_grad_per_block(warup_abs_grads,
                                                targeted_module_dims,
                                                block_dimension)

    selected_submatrix = _select_submatrix_based_on_grads(
        downsample_blocks_ratio, enable_analysis,
        layer_block_grad_analysis_path, num_total_blocks, block_means)

    return selected_submatrix


def _analyze_layer_level_grads(output_dir: str,
                               warup_abs_grads: LayerLevelBlockType,
                               mlp_or_attention: ModuleType) -> None:
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


def _select_submatrix_based_on_grads(downsample_blocks_ratio: float,
                                     enable_analysis: bool,
                                     layer_block_grad_analysis_path: str,
                                     num_total_blocks,
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
