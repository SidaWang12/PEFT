from functools import reduce
import math
from typing import Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from datasets import Dataset, load_dataset
from trl import SFTTrainer, TrlParser, ModelConfig, ScriptArguments, SFTConfig

from trainers.PeftTrainer import PeftTrainer
from helpers.monitoring import *
from helpers.logging import logger
from trainers.smt_gradient_helper import *


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
    tokenizer = _load_and_configure_tokenizer(model_args)
    model = _initialize_model(model_args.model_name_or_path, model_kwargs)
    datasets = _prepare_datasets(
        script_args.dataset_name,
        script_args.dataset_config,
        script_args.dataset_train_split
    )

    block_size = get_gcd_from_weight_shape(model)
    logger.info(f"block_size is {block_size}")

    # overfit_small_data = datasets["train"].select(range(100))
    # Initialize trainer
    trainer = PeftTrainer(
    # trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"], #.select(range(100, 200)),
        eval_dataset=datasets["test"], #.select(range(100, 200)),
        processing_class=tokenizer,
    )

    # Log initial memory stats
    TrainingMonitor.memory_stats()

    # Start training
    logger.info("Starting training...")
    trainer.train()

    analyze_layer_level_grads(training_args.output_dir, trainer.warmup_grads, trainer.warmup_grads_count)

    # Log final memory stats
    TrainingMonitor.memory_stats()
    logger.info("Training completed successfully")

    # logger.info("Saving the trained model...")
    # trainer.save_model()
    # logger.info("Finished saving the trained model...")


def get_gcd_from_weight_shape(model):
    all_dims = []
    for name, param in model.named_parameters():
        if "mlp" in name or "attn" in name:
            all_dims.extend(param.shape)
    return reduce(math.gcd, all_dims)


def _load_and_configure_tokenizer(
    model_args: ModelConfig,
) -> PreTrainedTokenizer:
    """Load and configure the tokenizer."""
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        fast_tokenizer=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _prepare_datasets(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    train_split: str = "train",
    test_size: float = 0.05, # TODO: adjust this parameter
    seed: int = 42
) -> Dict[str, Dataset]:
    """Load and split the dataset."""
    def preprocess_function(
        example: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"prompt": example["instruction"],
                "completion": example["answer"]}

    dataset = load_dataset(dataset_name, name=dataset_config)
    dataset = dataset[train_split]
    dataset = dataset.map(preprocess_function)
    return dataset.train_test_split(
        test_size=test_size,
        seed=seed,
        shuffle=True
    )

def _initialize_model(
    model_name: str,
    model_kwargs: Dict[str, Any]
) -> AutoModelForCausalLM:
    logger.info("loading model...")
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

if __name__ == "__main__":
    main()