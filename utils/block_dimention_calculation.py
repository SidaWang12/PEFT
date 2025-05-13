from transformers import AutoModelForCausalLM

from functools import reduce
import math


def calculate_block_dimension(model: AutoModelForCausalLM) -> int:
    """
    Computes the greatest common divisor (GCD) of all tensor dimensions
    in MLP and attention layers.
    """
    all_dims = []
    for name, param in model.named_parameters():
        if "mlp" in name or "attn" in name:
            all_dims.extend(param.shape)
    return reduce(math.gcd, all_dims)