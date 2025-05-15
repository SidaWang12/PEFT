import re

from transformers import AutoModelForCausalLM

from block_libs.get_module_names import get_module_name
from block_libs.types_and_structs import ModuleType, SelectedSubmatrixType


def model_freeze_unselected_matrix_layer(
    model: AutoModelForCausalLM,
    selected_mlp_parameters: SelectedSubmatrixType,
    selected_attention_parameters: SelectedSubmatrixType
) -> AutoModelForCausalLM:
    pattern = re.compile(r'model\.layers\.(\d+)\.')
    for name, param in model.named_parameters():
        if "mlp" in name:
            module_name = get_module_name(name, ModuleType.MLP)
            match = pattern.search(name)
            layer_number = int(match.group(1)) if match else None
            if (module_name, layer_number) in selected_mlp_parameters.keys():
                param.requires_grad = True
            else:
                param.requires_grad = False

        elif "self_attn" in name:
            module_name = get_module_name(name, ModuleType.ATTENTION)
            match = pattern.search(name)
            layer_number = int(match.group(1)) if match else None
            if (module_name,
                    layer_number) in selected_attention_parameters.keys():
                param.requires_grad = True

            else:
                param.requires_grad = False
        else:
            param.requires_grad = False

    return model
