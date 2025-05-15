from collections import defaultdict
import re

import torch
from tqdm import tqdm

from block_libs.get_module_names import get_module_name
from block_libs.types_and_structs import ModuleType


def prepare_batch(batch, tokenizer):
    full_texts = [p + a for p, a in zip(batch["prompt"], batch["completion"])]
    encodings = tokenizer(full_texts, padding=False, truncation=True)
    answer_encodings = tokenizer(batch["completion"],
                                 padding=False,
                                 truncation=True)
    max_len = max(len(ids) for ids in encodings["input_ids"])

    input_ids = []
    attention_mask = []
    labels = []

    for i in range(len(encodings["input_ids"])):
        seq_len = len(encodings["input_ids"][i])
        answer_len = len(answer_encodings["input_ids"][i])
        prompt_len = seq_len - answer_len

        pad_len = max_len - seq_len
        input_ids.append(encodings["input_ids"][i] +
                         [tokenizer.pad_token_id] * pad_len)
        attention_mask.append([1] * seq_len + [0] * pad_len)

        label = [-100] * prompt_len + answer_encodings["input_ids"][i] + [
            -100
        ] * pad_len
        labels.append(label)

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels)
    }


def compute_aggregated_saliency_batch(model, tokenizer, dataset, batch_size):
    model.eval()

    num_samples = len(dataset)
    saliency_dict = defaultdict(lambda: 0)
    device = model.device

    pattern = re.compile(r'model\.layers\.(\d+)\.')

    for batch_start in tqdm(range(0, num_samples, batch_size),
                            desc="Processing batches"):
        try:
            batch = dataset[batch_start:batch_start + batch_size]
            batch_data = prepare_batch(batch, tokenizer)

            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data["labels"].to(device)

            model.zero_grad()
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            outputs.loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None and ("mlp" in name
                                               or "self_attn" in name):
                    match = pattern.search(name)
                    layer_number = int(match.group(1)) if match else None
                    if "mlp" in name:
                        module_name = get_module_name(name, ModuleType.MLP)
                    else:
                        module_name = get_module_name(name,
                                                      ModuleType.ATTENTION)
                    key = (module_name, layer_number)

                    grad = param.grad.abs().detach().cpu()
                    saliency_dict[key] += grad

        except Exception as e:
            print(f"Error processing batch {batch_start}: {str(e)}")
            continue

    if num_samples > 0:
        for name in saliency_dict:
            if not isinstance(saliency_dict[name], int):
                saliency_dict[name] /= num_samples

    return saliency_dict
