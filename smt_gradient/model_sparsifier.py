import re

from transformers import AutoModelForCausalLM


def model_freeze_unselected_matrix_layer(model: AutoModelForCausalLM,
                                         select_parameters,
                                         select_attention_parameters):
    # selected_parameters: (module_name, layer_number, head_number)
    pattern = re.compile(r'model\.layers\.(\d+)\.')
    for name, param in model.named_parameters():
        if "mlp" in name:
            module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
            match = pattern.search(name)
            layer_number = int(match.group(1)) if match else None
            if (module_name, layer_number) in select_parameters.keys():
                param.requires_grad = True
                # print_rank_0(f"Layer set to grad = True:{name}",
                #              global_rank)

                # print("selected grad True layer")
                # print(module_name, layer_number)
            else:
                param.requires_grad = False
                # print_rank_0(f"Layer set to grad = Flase:{name}",
                #              global_rank)

        elif "self_attn" in name:
            # module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else None
            module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else 'o_proj' if 'o_proj' in name else None

            match = pattern.search(name)
            layer_number = int(match.group(1)) if match else None
            if (module_name,
                    layer_number) in select_attention_parameters.keys():
                param.requires_grad = True
                # print_rank_0(f"Layer set to grad = True:{name}",
                #              global_rank)

            else:
                param.requires_grad = False
                # print_rank_0(f"Layer set to grad = Flase:{name}",
                #              global_rank)

        else:
            param.requires_grad = False
            # print_rank_0(f"Layer set to grad = False:{name}", global_rank)

    return model
