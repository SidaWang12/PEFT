# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import random
import numpy as np
from transformers import set_seed, AutoTokenizer
import json
import os
import math
import torch
from transformers import AutoConfig
from transformers.integrations import HfDeepSpeedConfig # transformers.integrations under certain transformers version.

from deepspeed.accelerator import get_accelerator
from evaluation.helpers.model_names import LLAMA_8B_MODEL_NAMES

GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4

# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)
    else:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(msg)
        else:
            print(msg)

# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
# TODO(Hector, Sida): fix bug for llama3-8B, Llama3-8B use AutoTokenizer instead of LlamaTokenizer.
# TODO(Sida): look into QWen family tokenizers.
# rewrite the code for llama
def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    if model_name_or_path in ["yahma/llama-13b-hf", "NousResearch/Llama-2-13b-hf",
                              "yahma/llama-7b-hf", "meta-llama/Llama-2-7b-chat-hf"]:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path,
            fast_tokenizer=fast_tokenizer,
            add_bos_token=False)  # not adding start token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'right'
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            fast_tokenizer=fast_tokenizer,
            add_bos_token=False)  # not adding start token
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = 'right'
        if "DeepSeek-R1-Distill" not in model_name_or_path:
            tokenizer.padding_side = 'left'
            tokenizer.pad_token_id = 0
    return tokenizer


# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
def load_hf_tokenizer(model_name_or_path,
                      max_seq_len=2024,
                      fast_tokenizer=True,
                      add_special_tokens=None):

    if os.path.exists(model_name_or_path):
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file.get("_name_or_path",
                                             model_name_or_path)
            tokenizer = get_tokenizer(model_name,
                                      fast_tokenizer=fast_tokenizer)
    else:
        tokenizer = get_tokenizer(model_name_or_path,
                                  fast_tokenizer=fast_tokenizer)

    if add_special_tokens is not None:
        add_special_tokens = [add_special_tokens] if isinstance(add_special_tokens, str) \
            else add_special_tokens
        tokenizer.add_special_tokens(
            {'additional_special_tokens': add_special_tokens})

    tokenizer.model_max_length = max_seq_len
    return tokenizer

# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Source Code from: DeepSpeed example official website model_utils
# Refer to https://github.com/microsoft/DeepSpeedExamples/blob/75df1d7250452bcc7c065797a95c982bc8caab0b/applications/DeepSpeed-Chat/dschat/utils/model/model_utils.py#L19
def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


# Source Code from: DeepSpeed example official website model_utils
# Refer to https://github.com/microsoft/DeepSpeedExamples/blob/75df1d7250452bcc7c065797a95c982bc8caab0b/applications/DeepSpeed-Chat/dschat/utils/model/model_utils.py#L19


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    trained=False,
                    dropout=None):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    configure_dropout(model_config, dropout)

    # specially define llama3 model to fix the tokenizer issue in current hf version
    if model_name_or_path in LLAMA_8B_MODEL_NAMES or model_name_or_path == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
        or model_name_or_path == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
        tokenizer.pad_token_id = 0

    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if trained:
        model = model_class.from_config(model_config)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

    return model
