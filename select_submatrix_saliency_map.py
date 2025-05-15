from saliency_map.saliency_map_calculation.saliency_map_generator import compute_aggregated_saliency_batch
from saliency_map.saliency_map_calculation.saliency_map_plotter import plot_saliency_map
from trl import TrlParser, ModelConfig, ScriptArguments

from utils.monitoring import TrainingMonitor
from trl.trainer.sft_trainer import SFTConfig
from utils.logging_utils import logger
from utils.model_utils import load_and_configure_tokenizer, initialize_model, prepare_datasets


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
        "device_map": "auto"
    }
    logger.info("Model Kwargs: %s", model_kwargs)

    # Initialize components
    tokenizer = load_and_configure_tokenizer(model_args)
    model = initialize_model(model_args.model_name_or_path, model_kwargs)
    datasets = prepare_datasets(
        script_args.dataset_name,
        script_args.dataset_config,
        script_args.dataset_train_split,
        training_args.seed,
        test_set_percentage=0.2)

    # Log initial memory stats
    TrainingMonitor.memory_stats()
    train_data = datasets["train"]
    model.requires_grad_(True)

    saliency_scores = compute_aggregated_saliency_batch(
        model, tokenizer, train_data.select(range(0, 10)), batch_size=16)
    
    logger.info("Plotting saliency maps...")
    plot_saliency_map(training_args.output_dir,
                      saliency_scores)


if __name__ == "__main__":
    main()
