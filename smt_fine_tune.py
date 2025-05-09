import ast
import json
import os
from smt_gradient.model_sparsifier import model_freeze_unselected_matrix_layer
from trl import TrlParser, ModelConfig, ScriptArguments

from trainers.peft_trainer import PeftTrainer, PeftTrainerMode
from utils.monitoring import TrainingMonitor
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

    selected_submatrix = load_selected_submatrix(
        os.path.join(training_args.output_dir, 'selected_blocks.json'))
    logger.info(f"selected submatrix: {selected_submatrix}")

    model = model_freeze_unselected_matrix_layer(model, selected_submatrix, {})
    trainable_parameters_statistics(model)

    # overfit_small_data = datasets["train"].select(range(100))
    # Initialize trainer
    trainer = PeftTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],  #.select(range(100, 200)),
        eval_dataset=datasets["test"],  #.select(range(100, 200)),
        processing_class=tokenizer,
        mode=PeftTrainerMode.TrainingMode)

    # Log initial memory stats
    TrainingMonitor.memory_stats()

    # Start training
    logger.info("Starting training...")
    trainer.train()
    TrainingMonitor.memory_stats()
    logger.info("Training completed successfully")

    print_loss_through_whole_training(trainer.state.log_history)


def load_selected_submatrix(file_path: str) -> dict:
    with open(file_path, "r") as f:
        raw_dict = json.load(f)

    # Convert string keys like "('down_proj', 1)" back to tuple
    converted_dict = {ast.literal_eval(k): v for k, v in raw_dict.items()}
    return converted_dict


def trainable_parameters_statistics(model):
    logger.info("Print the requres_grad of model weight.")
    for name, param in model.named_parameters():
        logger.info(f"{name}, requres_grad, {param.requires_grad}")

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
