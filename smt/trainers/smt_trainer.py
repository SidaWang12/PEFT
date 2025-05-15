import re
import time
from typing import Any, Callable, Optional, Union
from libs.block_libs.get_module_names import get_module_name
from libs.block_libs.types_and_structs import LayerLevelBlockType, ModuleType
from trl.trainer.sft_trainer import SFTTrainer, SFTConfig
import torch
from torch import nn
from deepspeed.utils import safe_get_full_grad

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
            _get_warmup_mlp_grads(model, self.warmup_mlp_grads,
                                  self.downsample_mlp_blocks_ratio)
            _get_warmup_attention_grads(model, self.warmup_attention_grads,
                                        self.downsample_attention_blocks_ratio)
        self.sum_training_step_time += time.time() - start_time

        return loss


def _get_warmup_mlp_grads(model: torch.nn.Module,
                          warmup_mlp_grads: LayerLevelBlockType,
                          downsample_mlp_blocks_ratio: float) -> None:
    """
    Accumulate warmup gradients for MLP layers across steps.
    """
    pattern = re.compile(r'model\.layers\.(\d+)\.')

    for name, param in model.named_parameters():
        match = pattern.search(name)
        layer_number = int(match.group(1)) if match else None
        if downsample_mlp_blocks_ratio >= 0 and 'mlp' in name and 'weight' in name:
            grad = safe_get_full_grad(param)  # (hidden_dim, head_dim)
            module_name = get_module_name(name, ModuleType.MLP)
            key = (module_name, layer_number)

            if key not in warmup_mlp_grads:
                # warmup_mlp_grads[(module_name, layer_number)] = grad.detach().to(torch.float32)
                warmup_mlp_grads[key] = grad.detach().cpu().to(torch.float32)
            else:
                warmup_mlp_grads[key] += grad.detach().cpu().to(torch.float32)
                # warmup_mlp_grads[(module_name, layer_number)] += grad.detach().to(torch.float32)
                # del grad


def _get_warmup_attention_grads(
        model: torch.nn.Module, warmup_attention_grads: LayerLevelBlockType,
        downsample_attention_blocks_ratio: float) -> None:
    """
    Accumulate warmup gradients for attention layers across steps.
    """
    pattern = re.compile(r'model\.layers\.(\d+)\.')

    for name, param in model.named_parameters():
        match = pattern.search(name)
        layer_number = int(match.group(1)) if match else None
        if downsample_attention_blocks_ratio >= 0 and 'self_attn' in name and 'weight' in name:
            grad = safe_get_full_grad(param)  # (hidden_dim, head_dim)
            module_name = get_module_name(name, ModuleType.ATTENTION)
            key = (module_name, layer_number)

            if key not in warmup_attention_grads:
                warmup_attention_grads[key] = grad.detach().cpu().to(
                    torch.float32)

            else:
                warmup_attention_grads[key] += grad.detach().cpu().to(
                    torch.float32)
