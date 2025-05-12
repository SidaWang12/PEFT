from collections import defaultdict
import json
import os

from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from trl import TrlParser, ModelConfig, ScriptArguments

from utils.monitoring import GPUMemoryStatsCallback, TrainingMonitor
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling, SFTConfig
from utils.logging import logger
from utils.model_utils import load_and_configure_tokenizer, initialize_model, prepare_datasets, print_loss_through_whole_training
import seaborn as sns
from transformers import DataCollatorWithPadding


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
    model.requires_grad_(True)

    saliency_scores = compute_aggregated_saliency_batch(model, tokenizer, train_data.select(range(0, 2)))
    plot_saliency_map(training_args.output_dir, saliency_scores["model.layers.0.mlp.down_proj.weight"])


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

def prepare_batch(batch, tokenizer):
    full_texts = [p + a for p, a in zip(batch["prompt"], batch["completion"])]
    encodings = tokenizer(full_texts, padding=False, truncation=True)
    answer_encodings = tokenizer(batch["completion"], padding=False, truncation=True)
    max_len = max(len(ids) for ids in encodings["input_ids"])
    
    input_ids = []
    attention_mask = []
    labels = []
    
    for i in range(len(encodings["input_ids"])):
        seq_len = len(encodings["input_ids"][i])
        answer_len = len(answer_encodings["input_ids"][i])
        prompt_len = seq_len - answer_len
        
        pad_len = max_len - seq_len
        input_ids.append(encodings["input_ids"][i] + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append([1] * seq_len + [0] * pad_len)
        
        label = [-100] * prompt_len + answer_encodings["input_ids"][i] + [-100] * pad_len
        labels.append(label)
    
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels)
    }

def compute_aggregated_saliency_batch(
    model,
    tokenizer,
    dataset,
    batch_size=16,
    max_samples=None,
    target_layers=["down_proj"]
):
    """不使用DataCollator的显著性计算"""
    model.eval()
    
    # 1. 只对目标层启用梯度
    for name, param in model.named_parameters():
        param.requires_grad = any(layer in name for layer in target_layers)

    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    saliency_dict = defaultdict(lambda: 0)
    device = model.device

    for batch_start in tqdm(range(0, num_samples, batch_size), desc="Processing batches"):
        try:
            batch = dataset[batch_start : batch_start + batch_size]
            batch_data = prepare_batch(batch, tokenizer)
            
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data["labels"].to(device)
            
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            outputs.loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None and any(layer in name for layer in target_layers):
                    grad = param.grad.abs().detach().cpu()
                    saliency_dict[name] += grad
                    
        except Exception as e:
            print(f"Error processing batch {batch_start}: {str(e)}")
            continue
    
    if num_samples > 0:
        for name in saliency_dict:
            if not isinstance(saliency_dict[name], int):
                saliency_dict[name] /= num_samples
    
    return {k: v for k, v in saliency_dict.items() if not isinstance(v, int)}


if __name__ == "__main__":
    main()
