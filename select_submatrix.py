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

    # overfit_small_data = datasets["train"].select(range(100))
    # Initialize trainer
    trainer = PeftTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],  #.select(range(100, 200)),
        eval_dataset=datasets["test"],  #.select(range(100, 200)),
        processing_class=tokenizer,
        mode=PeftTrainerMode.SelectSubmatrixMode,
        callbacks=[GPUMemoryStatsCallback()])

    # Log initial memory stats
    TrainingMonitor.memory_stats()

    # Start training
    logger.info("Starting training...")
    trainer.train()
    TrainingMonitor.memory_stats()
    logger.info("Training completed successfully")

    logger.info("Selecting submatrix...")
    selected_mlp_submatrix = {}
    if training_args.downsample_mlp_blocks_ratio >= 0:
        selected_mlp_submatrix = select_submatrix(
            model, trainer.warmup_mlp_grads, trainer.state.global_step,
            training_args.enable_analysis, training_args.output_dir,
            training_args.downsample_mlp_blocks_ratio, SMTBlockType.MLP)
    logger.info(f"selected_mlp_submatrix {selected_mlp_submatrix}")

    selected_attention_submatrix = {}
    if training_args.downsample_attention_blocks_ratio >= 0:
        selected_attention_submatrix = select_submatrix(
            model, trainer.warmup_attention_grads, trainer.state.global_step,
            training_args.enable_analysis, training_args.output_dir,
            training_args.downsample_attention_blocks_ratio, SMTBlockType.ATTENTION)
    logger.info(f"selected_attention_submatrix {selected_attention_submatrix}")

    submatrix_file_path = os.path.join(training_args.output_dir,
                                       'selected_blocks.json')
    save_selected_submatrix(
        selected_mlp_submatrix=selected_mlp_submatrix,
        selected_attention_submatrix=selected_attention_submatrix,
        submatrix_file_path=submatrix_file_path)
    logger.info(f"Submatrix file is saved to {submatrix_file_path}")

    print_loss_through_whole_training(trainer.state.log_history)


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
