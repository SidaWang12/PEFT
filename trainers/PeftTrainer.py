from typing import Any, Callable, Optional, Union
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
from smt_gradient.smt_gradient_helper import *


class PeftTrainer(SFTTrainer):
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
        self.warmup_grads = {}

    def training_step(self,
                      model: nn.Module,
                      inputs: dict[str, Union[torch.Tensor, Any]],
                      num_items_in_batch=None) -> torch.Tensor:
        loss = super().training_step(model, inputs, num_items_in_batch)

        get_warmup_grads(model, self.warmup_grads)

        return loss
