import json
import os

from matplotlib import pyplot as plt
import torch
from trl import TrlParser, ModelConfig, ScriptArguments

from utils.monitoring import GPUMemoryStatsCallback, TrainingMonitor
from trl.trainer.sft_trainer import SFTConfig
from utils.logging import logger
from utils.model_utils import load_and_configure_tokenizer, initialize_model, prepare_datasets, print_loss_through_whole_training
import seaborn as sns


def main():
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    logger.info("Script Arguments: %s", script_args)
    logger.info("Training Arguments: %s", training_args)
    logger.info("Model Arguments: %s", model_args)

    # Model configuration
    model_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": model_args.torch_dtype,
    }
    logger.info("Model Kwargs: %s", model_kwargs)

    # Initialize components
    tokenizer = load_and_configure_tokenizer(model_args)
    model = initialize_model(model_args.model_name_or_path, model_kwargs)
    datasets = prepare_datasets(script_args.dataset_name,
                                script_args.dataset_config,
                                script_args.dataset_train_split,
                                training_args.seed,
                                # training_args.test_set_percentage)
                                test_set_percentage=0.2)

    # Log initial memory stats
    TrainingMonitor.memory_stats()
    train_data = datasets["train"]
    model.eval()
    model.requires_grad_(True)

    # saliency_scores = compute_saliency_map(model, tokenizer, train_data[0]["prompt"], train_data[0]["answer"])
    saliency_scores = compute_aggregated_saliency(model, tokenizer, train_data.select(range(0, 2)))
    plot_saliency_map(training_args.output_dir, saliency_scores)


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
    plt.savefig(os.path.join(output_dir, "a.jpg"))

def compute_saliency_map(model, tokenizer, prompt, answer):
    """
    Robust saliency map computation with length handling
    """
    # 1. Tokenize with same truncation for both
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
    )
    targets = tokenizer(
        answer, 
        return_tensors="pt", 
    )
    
    # 2. Align lengths by padding the shorter one
    seq_len = max(inputs["input_ids"].shape[1], targets["input_ids"].shape[1])
    inputs = tokenizer.pad(
        inputs, 
        padding="max_length", 
        max_length=seq_len,
        return_tensors="pt"
    )
    targets = tokenizer.pad(
        targets,
        padding="max_length",
        max_length=seq_len,
        return_tensors="pt"
    )
    
    # 3. Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    targets = {k: v.to(model.device) for k, v in targets.items()}
    
    # 4. Forward + backward
    model.zero_grad()
    outputs = model(**inputs, labels=targets["input_ids"])

    loss = outputs.loss
    loss.backward()
    
    down_proj_saliency = {}
    for name, param in model.named_parameters():
        if "down_proj" in name and param.requires_grad:
            down_proj_saliency = param.grad.detach().abs().cpu()
            break

    return down_proj_saliency

def compute_aggregated_saliency(model, tokenizer, train_data):
    """
    Accumulate saliency maps for all down_proj weights over entire training set.
    """
    model.eval()
    model.requires_grad_(True)

    aggregated_saliency = None
    num_samples = 0

    for sample in train_data:
        prompt, answer = sample["prompt"], sample["answer"]
        saliency = compute_saliency_map(model, tokenizer, prompt, answer)
        
        if aggregated_saliency is None:
            aggregated_saliency = saliency
        else:
            aggregated_saliency += saliency
        
        num_samples += 1
        print(num_samples)

    # Optional: average across samples
    aggregated_saliency /= num_samples

    return aggregated_saliency


if __name__ == "__main__":
    main()
