from typing import Any, Dict, Optional
from datasets import Dataset, load_dataset
from trl import TrlParser, ModelConfig, ScriptArguments

from helpers.logging import logger

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)

def load_and_configure_tokenizer(
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


def prepare_datasets(dataset_name: str, dataset_config: Optional[str],
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


def initialize_model(model_name: str,
                      model_kwargs: Dict[str, Any]) -> AutoModelForCausalLM:
    logger.info("loading model...")
    return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)