import json
import os
from utils.types_and_structs import SMTBlockType
from trl import TrlParser, ModelConfig, ScriptArguments

from trainers.peft_trainer import PeftTrainerMode, PeftTrainer
from utils.monitoring import GPUMemoryStatsCallback, TrainingMonitor
from utils.logging import logger
from smt_gradient.smt_gradient_selector import select_submatrix
from model_and_config_utils.peft_config import PeftConfig
from model_and_config_utils.model_utils import load_and_configure_tokenizer, initialize_model, prepare_datasets, print_loss_through_whole_training


def main():
    # Parse arguments
    parser = TrlParser((ScriptArguments, PeftConfig, ModelConfig))
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

    logger.info(
        f"downsample_mlp_blocks_ratio {training_args.downsample_mlp_blocks_ratio}"
    )
    logger.info(
        f"downsample_attention_blocks_ratio {training_args.downsample_attention_blocks_ratio}"
    )

    # Initialize components
    tokenizer = load_and_configure_tokenizer(model_args)
    model = initialize_model(model_args.model_name_or_path, model_kwargs)
    datasets = prepare_datasets(script_args.dataset_name,
                                script_args.dataset_config,
                                script_args.dataset_train_split,
                                training_args.seed,
                                training_args.test_set_percentage)

    # Log initial memory stats
    TrainingMonitor.memory_stats()
    train_data = datasets["train"]
    print(train_data.shape, train_data[0])
    model.eval()
    model.requires_grad_(True)

    saliency_scores, tokens = compute_saliency_map(model, tokenizer, train_data[0]["prompt"], train_data[0]["answer"])
    print("Token Saliency Scores:")
    for token, score in zip(tokens, saliency_scores):
        print(f"{token}: {score:.4f}")
    # print(len(saliency_scores))


def compute_saliency_map(model, tokenizer, prompt, answer):
    """
    Robust saliency map computation with length handling
    """
    # 1. Tokenize with same truncation for both
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        #max_length=tokenizer.model_max_length
    )
    targets = tokenizer(
        answer, 
        return_tensors="pt", 
        truncation=True, 
        #max_length=tokenizer.model_max_length
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
    
    # 5. Get saliency (absolute gradient of input embeddings)
    embed_grad = model.get_input_embeddings().weight.grad
    saliency = embed_grad[inputs["input_ids"]].abs().sum(dim=-1).squeeze()
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-9)
    
    # Convert to tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    
    return saliency.tolist(), tokens


def save_selected_submatrix(selected_mlp_submatrix,
                            selected_attention_submatrix, submatrix_file_path):
    combined = {
        "selected_mlp_submatrix":
        {str(k): v
         for k, v in selected_mlp_submatrix.items()},
        "selected_attention_submatrix":
        {str(k): v
         for k, v in selected_attention_submatrix.items()}
    }

    with open(submatrix_file_path, "w") as f:
        json.dump({str(k): v
                   for k, v in combined.items()},
                  f,
                  separators=(",", ":"),
                  indent=None)


if __name__ == "__main__":
    main()
