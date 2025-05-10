import os
from typing import Dict, List

from matplotlib import pyplot as plt
import numpy as np

from utils.types import LayerLevelGradType


def plot_layer_level_grads(grad_statistics: LayerLevelGradType,
                           output_dir: str, mlp_or_attention: str,
                           statistical_method: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Extract layer numbers and gradients for each projection type
    layers = sorted(set(layer for (_, layer) in grad_statistics.keys()))

    # Prepare data for plotting
    if mlp_or_attention == "mlp":
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
