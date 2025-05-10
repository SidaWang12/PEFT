import re

from transformers import AutoModelForCausalLM

from utils.types import SelectedSubmatrixType


def model_freeze_unselected_matrix_layer(
        model: AutoModelForCausalLM, selected_mlp_parameters: SelectedSubmatrixType,
        selected_attention_parameters: SelectedSubmatrixType) -> AutoModelForCausalLM:
    pattern = re.compile(r'model\.layers\.(\d+)\.')
    for name, param in model.named_parameters():
        if "mlp" in name:
            module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
            match = pattern.search(name)
            layer_number = int(match.group(1)) if match else None
            if (module_name, layer_number) in selected_mlp_parameters.keys():
                param.requires_grad = True
            else:
                param.requires_grad = False

        elif "self_attn" in name:
            # module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else None
            module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else 'o_proj' if 'o_proj' in name else None

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
