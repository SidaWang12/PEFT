import logging
from typing import Dict, Any, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    TrainerCallback
)
from datasets import Dataset, load_dataset
from trl import SFTTrainer, TrlParser, ModelConfig, ScriptArguments, SFTConfig

from helpers.monitoring import *
from helpers.logging import logger


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
    tokenizer = load_and_configure_tokenizer(model_args, training_args)
    model = initialize_model(model_args.model_name_or_path, model_kwargs)
    datasets = prepare_datasets(
        script_args.dataset_name,
        script_args.dataset_config,
        script_args.dataset_train_split
    )

    # overfit_small_data = datasets["train"].select(range(100))
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        processing_class=tokenizer,
        callbacks=[MemoryStatsCallback(), LossLoggingCallback()],
    )

    # Log initial memory stats
    TrainingMonitor.memory_stats()

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Log final memory stats
    TrainingMonitor.memory_stats()
    logger.info("Training completed successfully")

def load_and_configure_tokenizer(
    model_args: ModelConfig,
    training_args: SFTConfig
) -> PreTrainedTokenizer:
    """Load and configure the tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            fast_tokenizer=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        raise


def prepare_datasets(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    train_split: str = "train",
    test_size: float = 0.2,
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

def initialize_model(
    model_name: str,
    model_kwargs: Dict[str, Any]
) -> AutoModelForCausalLM:
    """Initialize the model with given configuration."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

if __name__ == "__main__":
    main()