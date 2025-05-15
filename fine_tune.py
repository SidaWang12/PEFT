from transformers import AutoModelForCausalLM
import json
import os
from smt.model_sparsifier.unselected_block_freezer import convert_linear_layer_to_matrix_sparsity
from block_libs.types_and_structs import SelectedSubmatrixType
from smt.model_sparsifier.unselected_layer_freezer import model_freeze_unselected_matrix_layer
from trl import TrlParser, ModelConfig, ScriptArguments
from transformers.trainer_callback import TrainerState
from smt.trainers.smt_trainer import SMTTrainer, SMTTrainerMode
from utils.monitoring import GPUMemoryStatsCallback, TrainingMonitor
from utils.logging_utils import logger, log_training_metrics
from peft_config.peft_config import PeftConfig
from utils.model_utils import load_and_configure_tokenizer, initialize_model, prepare_datasets
from block_libs.block_dimention_calculation import calculate_block_dimension

from deepspeed.profiling.flops_profiler import FlopsProfiler


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
        "device_map": "auto"
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

    selected_mlp_submatrix, selected_attention_submatrix = _load_selected_submatrix(
        os.path.join(training_args.output_dir, 'selected_blocks.json'))
    logger.info(f"selected_mlp_submatrix: {selected_mlp_submatrix}")
    logger.info(
        f"selected_attention_submatrix: {selected_attention_submatrix}")

    model = model_freeze_unselected_matrix_layer(model, selected_mlp_submatrix,
                                                 selected_attention_submatrix)

    logger.info("Print the requres_grad of model weight.")
    for name, param in model.named_parameters():
        logger.info(f"{name}, requres_grad, {param.requires_grad}")

    logger.info(
        "Trainable parameters statistics after freezing unselected layers:")
    _trainable_parameters_statistics(model)

    block_dimension = calculate_block_dimension(model)
    model = convert_linear_layer_to_matrix_sparsity(
        model, selected_mlp_submatrix, selected_attention_submatrix,
        block_dimension)

    flops_profiler = FlopsProfiler(model)
    flops_profiler.start_profile()

    logger.info(
        "Trainable parameters statistics after freezing unselected blocks:")
    selected_param_num = _trainable_parameters_statistics(model)

    if selected_param_num == 0:
        logger.info("Trainable parameters number is 0. Skip training.")
        return

    # Initialize trainer
    trainer = SMTTrainer(model=model,
                         args=training_args,
                         train_dataset=datasets["train"],
                         eval_dataset=datasets["test"],
                         processing_class=tokenizer,
                         mode=SMTTrainerMode.TrainingMode,
                         callbacks=[GPUMemoryStatsCallback()])

    # Log initial memory stats
    TrainingMonitor.memory_stats()

    # Start training
    logger.info("Starting training...")
    trainer.train()
    TrainingMonitor.memory_stats()
    logger.info("Training completed successfully")

    flops_profiler.stop_profile()
    log_training_metrics(trainer.state, trainer.sum_training_step_time,
                         flops_profiler)

def _load_selected_submatrix(selected_submatrix_file: str) -> \
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


def _trainable_parameters_statistics(model: AutoModelForCausalLM) -> int:
    # print matrix sparsity trainable parameters
    total_num = sum(p.numel() for p in model.parameters())
    selected_num = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    logger.info(f"Number of Total parameters: {total_num/1e6} M")
    rate = (selected_num / total_num) * 100
    logger.info(f"Number of trainable parameters: {selected_num/1e6} M,\ns \
               about {rate}% matrix sparsity parameters in the model are training"
                )
    return selected_num


if __name__ == "__main__":
    main()
