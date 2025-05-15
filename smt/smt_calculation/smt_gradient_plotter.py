import glob
import os
import subprocess
from typing import Dict, List

from matplotlib import pyplot as plt
import numpy as np
import torch
import seaborn as sns

from block_libs.types_and_structs import LayerLevelBlockType, ModuleType


def generate_grad_heatmaps(grads_heatmap_path: str,
                           warup_abs_grads: LayerLevelBlockType) -> None:
    def downsample(tensor, new_h, new_w):
        h, w = tensor.shape
        tensor = tensor[:h - h % new_h, :w - w % new_w]
        tensor = tensor.view(new_h, h // new_h, new_w,
                             w // new_w).mean(3).mean(1)
        return tensor

    proj_names = set()
    for key in warup_abs_grads:
        proj_names.add(key[0])
    proj_names = sorted(proj_names)

    downsampled_data = {}
    for key in warup_abs_grads:
        grads = warup_abs_grads[key]

        vmin, vmax = float('inf'), float('-inf')

    for key in warup_abs_grads:
        grads = warup_abs_grads[key]
        down = downsample(grads, 100, 100)
        downsampled_data[key] = down

    layer_to_projections = {}
    for key, down in downsampled_data.items():
        module, layer = key
        if layer not in layer_to_projections:
            layer_to_projections[layer] = {}
        layer_to_projections[layer][module] = down

    # Define consistent color scale across all plots
    all_values = torch.cat([v.flatten() for v in downsampled_data.values()])
    vmin, vmax = all_values.min().item(), all_values.max().item()

    # Plot each layer with 3 projections side by side
    for layer_idx, module_data in layer_to_projections.items():
        _, axes = plt.subplots(1, len(proj_names),
                               figsize=(18, 6))  # one row, 3 columns

        for i, proj in enumerate(proj_names):
            ax = axes[i]
            data = module_data.get(proj)

            if data is not None:
                sns.heatmap(data.numpy(),
                            ax=ax,
                            cmap='viridis',
                            vmin=vmin,
                            vmax=vmax)
                ax.set_title(f"{proj}, layer {layer_idx}")
            else:
                ax.axis("off")  # Hide subplot if projection is missing

            ax.set_xlabel("Column Index")
            ax.set_ylabel("Row Index")

        plt.tight_layout()
        save_path = os.path.join(grads_heatmap_path,
                                 f"layer_{layer_idx}_all_proj.jpg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    output_video = os.path.join(grads_heatmap_path,
                                f"{proj_names} grads_heatmap.mp4")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "2",
        "-pattern_type",
        "glob",
        "-i",
        os.path.join(grads_heatmap_path, "layer_*.jpg"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "mjpeg",  # Use MJPEG codec
        "-q:v",
        "2",  # Quality (lower is better)
        output_video.replace(".mp4", ".avi")
    ]

    subprocess.run(ffmpeg_cmd, check=True)

    for file_path in glob.glob(os.path.join(grads_heatmap_path, "*.jpg")):
        os.remove(file_path)


def plot_layer_level_grads(grad_statistics: LayerLevelBlockType,
                           output_dir: str, mlp_or_attention: ModuleType,
                           statistical_method: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Extract layer numbers and gradients for each projection type
    layers = sorted(set(layer for (_, layer) in grad_statistics.keys()))

    # Prepare data for plotting
    if mlp_or_attention == ModuleType.MLP:
        proj_names = ['gate_proj', 'up_proj', 'down_proj']
        colors = ['blue', 'green', 'red']
    else:
        proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        colors = ['blue', 'green', 'red', 'black']

    # Create figure with 3 subplots
    _, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for proj_name, color, ax in zip(proj_names, colors, axes):
        grads = [grad_statistics[(proj_name, layer)] for layer in layers]
        ax.plot(layers, grads, 'o-', color=color)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Gradient ' + statistical_method)
        ax.set_title(
            f'{proj_name} {statistical_method} gradient across layers')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'{statistical_method}_grad_per_layer.jpg'))


def plot_gradient_per_block_distribution(
        analysis_plot_path: str,
        gradients_per_block: Dict[str, List[float]]) -> None:
    all_grads = np.concatenate(list(gradients_per_block.values()))
    global_min = np.min(all_grads)
    global_max = np.max(all_grads)

    num_modules = len(gradients_per_block)
    _, axes = plt.subplots(num_modules, 1, figsize=(10, 4 * num_modules))

    # If only one module, wrap axes in a list
    if num_modules == 1:
        axes = [axes]

    # Plot each module's gradient distribution
    for ax, (module_name, grads) in zip(axes, gradients_per_block.items()):
        # Convert to numpy array for processing
        grads = np.array(grads)

        # Plot histogram
        ax.hist(grads, bins=150, alpha=0.7, color='blue')
        ax.set_title(f'Gradient Distribution: {module_name}')
        ax.set_xlabel('Gradient Magnitude')
        ax.set_ylabel('Frequency')

        ax.set_xlim(global_min * 0.9, global_max * 1.1)

        # Add vertical line at mean
        mean_grad = np.mean(grads)
        ax.axvline(mean_grad,
                   color='red',
                   linestyle='--',
                   label=f'Mean: {mean_grad:.2e}')

        # Add statistics text
        stats_text = (f'Min: {np.min(grads):.2e}\n'
                      f'Max: {np.max(grads):.2e}\n'
                      f'Std: {np.std(grads):.2e}')
        ax.text(0.98,
                0.85,
                stats_text,
                transform=ax.transAxes,
                ha='right',
                va='top',
                bbox=dict(facecolor='white', alpha=0.5))

        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(analysis_plot_path, 'gradient_per_block.jpg'))
