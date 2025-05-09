import json
import os
from typing import Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from datasets import Dataset, load_dataset
from trl import TrlParser, ModelConfig, ScriptArguments

from trainers.PeftTrainer import PeftTrainer
from helpers.monitoring import *
from helpers.logging import logger
from smt_gradient.smt_gradient_helper import *
from trainers.peft_config import PeftConfig


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
    tokenizer = _load_and_configure_tokenizer(model_args)
    model = _initialize_model(model_args.model_name_or_path, model_kwargs)
    datasets = _prepare_datasets(script_args.dataset_name,
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

    # Start training
    logger.info("Starting training...")
    trainer.train()
    TrainingMonitor.memory_stats()
    logger.info("Training completed successfully")

    selected_submatrix = process_and_select_submatrix(model, trainer.warmup_grads, trainer.state.global_step, training_args.enable_analysis,
      training_args.output_dir, training_args.downsample_attention_blocks_ratio)
    logger.info(f"selected_ranked_block {selected_submatrix}")
    
    with open(os.path.join(training_args.output_dir, 'selected_blocks.json'),
              "w") as f:
        json.dump({str(k): v
                   for k, v in selected_submatrix.items()},
                  f,
                  separators=(",", ":"),
                  indent=None)


def _load_and_configure_tokenizer(
        model_args: ModelConfig) -> PreTrainedTokenizer:
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


def _prepare_datasets(dataset_name: str, dataset_config: Optional[str],
                      train_split: str, seed,
                      test_set_percentage: float) -> Dict[str, Dataset]:
    """Load and split the dataset."""
    def preprocess_function(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prompt": example["instruction"],
            "completion": example["answer"]
        }

    dataset = load_dataset(dataset_name, name=dataset_config)
    dataset = dataset[train_split]
    dataset = dataset.map(preprocess_function)
    return dataset.train_test_split(test_size=test_set_percentage,
                                    seed=seed,
                                    shuffle=True)


def _initialize_model(model_name: str,
                      model_kwargs: Dict[str, Any]) -> AutoModelForCausalLM:
    logger.info("loading model...")
    return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)


if __name__ == "__main__":
    main()
