import re
from libs.block_libs.get_module_names import get_module_name
from libs.block_libs.types_and_structs import LayerLevelBlockType, ModuleType
import torch
from deepspeed.utils import safe_get_full_grad

def get_warmup_mlp_grads(model: torch.nn.Module,
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


def get_warmup_attention_grads(
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