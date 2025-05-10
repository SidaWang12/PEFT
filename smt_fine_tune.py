from transformers import AutoModelForCausalLM
import json
import os
from utils.types_and_structs import SelectedSubmatrixType
from smt_gradient.model_sparsifier import model_freeze_unselected_matrix_layer
from trl import TrlParser, ModelConfig, ScriptArguments

from trainers.peft_trainer import PeftTrainer, PeftTrainerMode
from utils.monitoring import GPUMemoryStatsCallback, TrainingMonitor
from utils.logging import logger
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

    # Initialize components
    tokenizer = load_and_configure_tokenizer(model_args)
    datasets = prepare_datasets(script_args.dataset_name,
                                script_args.dataset_config,
                                script_args.dataset_train_split,
                                training_args.seed,
                                training_args.test_set_percentage)

    model = initialize_model(model_args.model_name_or_path, model_kwargs)

    selected_mlp_submatrix, selected_attention_submatrix = load_selected_submatrix(
        os.path.join(training_args.output_dir, 'selected_blocks.json'))
    logger.info(f"selected_mlp_submatrix: {selected_mlp_submatrix}")
    logger.info(
        f"selected_attention_submatrix: {selected_attention_submatrix}")

    model = model_freeze_unselected_matrix_layer(model, selected_mlp_submatrix,
                                                 selected_attention_submatrix)

    logger.info("Print the requres_grad of model weight.")
    for name, param in model.named_parameters():
        logger.info(f"{name}, requres_grad, {param.requires_grad}")

    trainable_parameters_statistics(model)

    # overfit_small_data = datasets["train"].select(range(100))
    # Initialize trainer
    trainer = PeftTrainer(model=model,
                          args=training_args,
                          train_dataset=datasets["train"],
                          eval_dataset=datasets["test"],
                          processing_class=tokenizer,
                          mode=PeftTrainerMode.TrainingMode,
                          callbacks=[GPUMemoryStatsCallback()])

    # Log initial memory stats
    TrainingMonitor.memory_stats()

    # Start training
    logger.info("Starting training...")
    trainer.train()
    TrainingMonitor.memory_stats()
    logger.info("Training completed successfully")

    print_loss_through_whole_training(trainer.state.log_history)


def load_selected_submatrix(selected_submatrix_file: str) -> \
        tuple[SelectedSubmatrixType, SelectedSubmatrixType]:
    """Load two submatrices from a JSON file"""
    with open(selected_submatrix_file, 'r') as f:
        data = json.load(f)

    # Convert keys back to original type if they were non-string keys
    selected_mlp_submatrix = {
        eval(k): v
        for k, v in data["selected_mlp_submatrix"].items()
    }
    selected_attention_submatrix = {
        eval(k): v
        for k, v in data["selected_attention_submatrix"].items()
    }

    return selected_mlp_submatrix, selected_attention_submatrix


def trainable_parameters_statistics(model: AutoModelForCausalLM):
    # print matrix sparsity trainable parameters
    total_num = sum(p.numel() for p in model.parameters())
    selected_num = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    logger.info(f"Number of Total parameters: {total_num/1e6} M")
    rate = (selected_num / total_num) * 100
    logger.info(f"Number of trainable parameters: {selected_num/1e6} M,\ns \
               about {rate}% matrix sparsity parameters in the model are training"
                )


if __name__ == "__main__":
    main()
