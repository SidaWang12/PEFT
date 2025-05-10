
def get_module_name(name: str, mlp_or_attention: str) -> str:
    if mlp_or_attention == "attention":
        PROJ_MAPPING = {
                'q_proj': 'q_proj',
                'k_proj': 'k_proj',
                'v_proj': 'v_proj',
                'o_proj': 'o_proj'
            }
    else:
        PROJ_MAPPING = {
            'gate_proj': 'gate_proj',
            'up_proj': 'up_proj',
            'down_proj': 'down_proj',
        }

    module_name = next((proj for proj in PROJ_MAPPING if proj in name),
                        None)
    if module_name is None:
        raise ValueError(
            f"Layer name '{name}' must contain one of {sorted(PROJ_MAPPING)}. "
            "Check your layer naming convention.")
    
    return module_name