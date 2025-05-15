import time
from typing import Any, Callable, Optional, Union
from smt.smt_calculation.get_warmup_grads import get_warmup_attention_grads, get_warmup_mlp_grads
from trl.trainer.sft_trainer import SFTTrainer, SFTConfig
import torch
from torch import nn

from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from datasets import Dataset, IterableDataset

from enum import Enum, auto


class SMTTrainerMode(Enum):
    SelectSubmatrixMode = auto()
    TrainingMode = auto()


class SMTTrainer(SFTTrainer):
    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        args: Optional[Union[SFTConfig, TrainingArguments]] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase,
                                         BaseImageProcessor,
                                         FeatureExtractionMixin,
                                         ProcessorMixin]] = None,
        mode: SMTTrainerMode = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer],
                          Optional[torch.optim.lr_scheduler.LambdaLR]] = (
                              None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer],
                                                 dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Union[Callable[[dict], str],
                                        Callable[[dict], list[str]]]] = None,
    ):
        super().__init__(model, args, data_collator, train_dataset,
                         eval_dataset, processing_class, compute_loss_func,
                         compute_metrics, callbacks, optimizers,
                         optimizer_cls_and_kwargs,
                         preprocess_logits_for_metrics, peft_config,
                         formatting_func)
        self.warmup_mlp_grads = {}
        self.warmup_attention_grads = {}
        self.mode = mode
        self.downsample_mlp_blocks_ratio = args.downsample_mlp_blocks_ratio
        self.downsample_attention_blocks_ratio = args.downsample_attention_blocks_ratio
        self.sum_training_step_time = 0.0

    def training_step(self,
                      model: nn.Module,
                      inputs: dict[str, Union[torch.Tensor, Any]],
                      num_items_in_batch=None) -> torch.Tensor:
        start_time = time.time()
        loss = super().training_step(model, inputs, num_items_in_batch)

        if self.mode == SMTTrainerMode.SelectSubmatrixMode:
            get_warmup_mlp_grads(model, self.warmup_mlp_grads,
                                  self.downsample_mlp_blocks_ratio)
            get_warmup_attention_grads(model, self.warmup_attention_grads,
                                        self.downsample_attention_blocks_ratio)
        self.sum_training_step_time += time.time() - start_time

        return loss

