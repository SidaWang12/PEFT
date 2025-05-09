from trl import TrlParser, ModelConfig, ScriptArguments

from trainers.peft_trainer import PeftTrainer
from helpers.monitoring import *
from helpers.logging import logger
from smt_gradient.smt_gradient_helper import *
from helpers.peft_config import PeftConfig
from helpers.model_utils import *

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
    )

    # Log initial memory stats
    TrainingMonitor.memory_stats()


if __name__ == "__main__":
    main()
