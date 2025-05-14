import re

import torch
from torch import nn

from smt.model_sparsifier.sparse_linear import BlockSparseLinear


def convert_linear_layer_to_matrix_sparsity(model,
                                            selected_mlp_submatrix,
                                            selected_attention_submatrix,
                                            block_dimension,
                                            part_module_name=['.layers']):
    pattern = re.compile(r'model\.layers\.(\d+)\.')

    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(part in name
                                                 for part in part_module_name):
            replace_name.append(name)

    for name in replace_name:
        if "mlp" in name:
            module = _recursive_getattr(model, name)
            if module.weight.requires_grad:
                module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
                match = pattern.search(name)
                layer_number = int(match.group(1)) if match else None

                if (module_name,
                        layer_number) in selected_mlp_submatrix.keys():
                    selected_blocks_list = selected_mlp_submatrix[(
                        module_name, layer_number)]
                    tmp = BlockSparseLinear(
                        module.weight,
                        bias=None,
                        selected_blocks_list=selected_blocks_list,
                        block_dimension=block_dimension).to(
                            module.weight.device).to(module.weight.dtype)

                    recursive_setattr(model, name, tmp)
        if "self_attn" in name:
            module = _recursive_getattr(model, name)
            if module.weight.requires_grad:
                module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else 'o_proj' if 'o_proj' in name else None
                match = pattern.search(name)
                layer_number = int(match.group(1)) if match else None

                if (module_name,
                        layer_number) in selected_attention_submatrix.keys():
                    selected_blocks_list = selected_attention_submatrix[(
                        module_name, layer_number)]
                    tmp = BlockSparseLinear(
                        module.weight,
                        bias=None,
                        selected_blocks_list=selected_blocks_list,
                        block_dimension=block_dimension).to(
                            module.weight.device).to(module.weight.dtype)

                    recursive_setattr(model, name, tmp)
        # TODO: support attention.

    return model


def recursive_setattr(model, module_name, module):
    """
    Recursively set the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to set the attribute in.
        module_name (`str`)
            The name of the module to set the attribute in.
        module (`torch.nn.Module`)
            The module to set the attribute to.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)


def _recursive_getattr(model, module_name):
    """
    Recursively get the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to get the attribute from.
        module_name (`str`)
            The name of the module to get the attribute from.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output
