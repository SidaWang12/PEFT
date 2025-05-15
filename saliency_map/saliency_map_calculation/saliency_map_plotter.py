from matplotlib import pyplot as plt
import torch
import seaborn as sns
import os


def plot_saliency_map(output_dir, saliency_scores):
    def _downsample_tensor(tensor: torch.Tensor, new_h: int,
                          new_w: int) -> torch.Tensor:
        h, w = tensor.shape
        tensor = tensor[:h - h % new_h, :w -
                        w % new_w]  # crop to divisible shape
        return tensor.view(new_h, h // new_h, new_w,
                           w // new_w).mean(3).mean(1)
    save_dir = os.path.join(output_dir, "saliency_map")
    os.makedirs(save_dir, exist_ok=True)

    # First pass: find global min/max for consistent scaling
    global_min = float('inf')
    global_max = -float('inf')
    
    downsampled_maps = {}
    for key, saliency_map in saliency_scores.items():
        downsampled = _downsample_tensor(saliency_map, 100, 100)
        downsampled_maps[key] = downsampled
        current_min = downsampled.min().item()
        current_max = downsampled.max().item()
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)

    # Second pass: plot all maps with same scale
    for key, downsampled in downsampled_maps.items():
        plt.figure(figsize=(12, 10))
        sns.heatmap(downsampled.to(torch.float32).numpy(), 
                   cmap='viridis',
                   vmin=global_min,
                   vmax=global_max,
                   cbar_kws={'label': 'Saliency Intensity'})
        plt.title(f"Saliency Map: {key}")
        plt.xlabel("Col Index")
        plt.ylabel("Row Index")
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{key}.jpg"), 
                   bbox_inches='tight', 
                   dpi=300)
        plt.close()

