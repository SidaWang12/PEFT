

from matplotlib import pyplot as plt
import torch
import seaborn as sns
import os


def plot_saliency_map(output_dir, saliency_scores):
    def downsample_tensor(tensor: torch.Tensor, new_h: int, new_w: int) -> torch.Tensor:
        h, w = tensor.shape
        tensor = tensor[:h - h % new_h, :w - w % new_w]  # crop to divisible shape
        return tensor.view(new_h, h // new_h, new_w, w // new_w).mean(3).mean(1)

    # Downsample to 100x100 for visualization
    downsampled = downsample_tensor(saliency_scores, 100, 100)

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(downsampled.to(torch.float32).numpy(), cmap='viridis')
    plt.title("Saliency Map")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "saliency_map.jpg"))
