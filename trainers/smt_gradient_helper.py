

import re
import os
from deepspeed.utils import safe_get_full_grad
import torch
import matplotlib.pyplot as plt


def get_warmup_grads(model, warmup_grads, warmup_grads_count):
    pattern = re.compile(r'model\.layers\.(\d+)\.')

    for name, param in model.named_parameters():
        match = pattern.search(name)
        layer_number = int(match.group(1)) if match else None
        if 'mlp' in name:
            grad = safe_get_full_grad(
                param)  # (hidden_dim, head_dim)
            module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
            key = (module_name, layer_number)

            #defaultdict(torch.float32)
            if key not in warmup_grads:
                # warmup_grads[(module_name, layer_number)] = grad.detach().to(torch.float32)
                warmup_grads[key] = grad.detach().cpu().to(
                        torch.float32)
                warmup_grads_count[key] = 1

            else:
                warmup_grads[key] += grad.detach().cpu().to(
                        torch.float32)
                warmup_grads_count[key] += 1
                # warmup_grads[(module_name, layer_number)] += grad.detach().to(torch.float32)
                # del grad

def _plot_layer_level_grads(grad_statistics, output_dir, statistical_method):
    # Extract layer numbers and gradients for each projection type
    layers = sorted(set(layer for (_, layer) in grad_statistics.keys()))
    gate_grads = [grad_statistics[('gate_proj', layer)] for layer in layers]
    up_grads = [grad_statistics[('up_proj', layer)] for layer in layers]
    down_grads = [grad_statistics[('down_proj', layer)] for layer in layers]

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot gate_proj gradients
    ax1.plot(layers, gate_grads, 'o-', color='blue')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Gradient Magnitude')
    ax1.set_title('gate_proj ' + statistical_method + ' gradient across layers')
    ax1.grid(True)

    # Plot up_proj gradients
    ax2.plot(layers, up_grads, 'o-', color='green')
    ax1.set_xlabel('Layer Index')
    ax2.set_ylabel('Gradient Magnitude')
    ax2.set_title('up_proj '  + statistical_method + ' gradient across layers')
    ax2.grid(True)

    # Plot down_proj gradients
    ax3.plot(layers, down_grads, 'o-', color='red')
    ax1.set_xlabel('Layer Index')
    ax3.set_ylabel('Gradient Magnitude')
    ax3.set_title('down_proj '  + statistical_method + ' gradient across layers')
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, statistical_method + '_grad_per_layer.jpg'))


def analyze_layer_level_grads(output_dir, warmup_grads, warmup_grads_count):
    grad_statistics = {}
    for key in warmup_grads:
        grad_statistics[key] = warmup_grads[key] / warmup_grads_count[key]
        grad_statistics[key] = grad_statistics[key].abs().mean().item()
    _plot_layer_level_grads(grad_statistics, output_dir, "mean")

    grad_statistics = {}
    for key in warmup_grads:
        grad_statistics[key] = warmup_grads[key] / warmup_grads_count[key]
        grad_statistics[key] = grad_statistics[key].var().mean().item()
    _plot_layer_level_grads(grad_statistics, output_dir, "var")

    

