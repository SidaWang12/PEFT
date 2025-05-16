'''
Example command:
accelerate launch \
  --main_process_port 64941 \
  evaluation/eval.py \
  --data_path /home/sidaw/Projects/llm/llm_dataset/LLM-Adapters/dataset \
  --model_name_or_path openlm-research/open_llama_3b \
  --tokenizer_path openlm-research/open_llama_3b \
  --per_device_eval_batch_size 4 \
  --seed 1234 \
  --dtype bf16 \
  --dataset boolq piqa social_i_qa ARC-Challenge ARC-Easy openbookqa hellaswag winogrande \
  --output_dir tmp
'''

# Source code modified from 
# 1. https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/multi_dataset_eval.py;
# 2.https://github.com/aksh555/LoRA-Soups/blob/main/evaluate.py; 
# 3. https://github.com/aksh555/LoRA-Soups/blob/main/utils.py
# 4. https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/commonsense_evaluate.py;

import tqdm
import argparse
import os
import sys
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import re
import json
import torch
from transformers import StoppingCriteria


from evaluation.helpers.eval_helpers import (
    print_rank_0,
    set_random_seed,
    create_hf_model,
    load_hf_tokenizer,
)

from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
)
import argparse
from accelerate import Accelerator
from accelerate.utils import gather_object


i_prompt = '''<s> Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Response:
'''


def extract_answer(dataset, sentence: str) -> float:
    sentence = sentence.lower()
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5',
                                  sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4',
                                  sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(
            stop_id_sequences[0],
            list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist(
                ) == stop_sequence:
                    sequences_should_be_stopped.append(True)
                    break
            sequences_should_be_stopped.append(False)
        return all(sequences_should_be_stopped)

@torch.no_grad()
def generate_completions(model,
                         device,
                         tokenizer,
                         prompts,
                         batch_size=1,
                         stop_id_sequences=None,
                         disable_tqdm=False,
                         verbose=False,
                         **generation_kwargs):
    generations = []
    if hasattr(model, "module"):
        print_rank_0(f'-----{model.module.generation_config}-----')
    else:
        print_rank_0(f'-----{model.generation_config}-----')

    if generation_kwargs:
        print_rank_0(f'-----{generation_kwargs}-----')

    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokenized_prompts = tokenizer(batch_prompts,
                                      padding='longest',
                                      return_tensors="pt")
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        batch_input_ids = batch_input_ids.to(device)
        attention_mask = attention_mask.to(device)

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)]
                if stop_id_sequences else None,
                **generation_kwargs)
            batch_outputs = batch_outputs.detach().cpu()

            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1],
                                           batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx, token_idx:token_idx +
                                             len(stop_sequence)].tolist() ==
                               stop_sequence
                               for stop_sequence in stop_id_sequences):
                            batch_outputs[output_idx,
                                          token_idx:] = tokenizer.pad_token_id
                            break

            # in case piece id out of range
            batch_outputs[
                batch_outputs >= tokenizer.vocab_size] = tokenizer.unk_token_id
            batch_outputs[batch_outputs == -1] = tokenizer.unk_token_id

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs,
                                                   skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids,
                                                   skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [
                prompt for prompt in batch_prompts
                for _ in range(num_return_sequences)
            ]
            batch_generations = [
                output[len(prompt):]
                for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""
                                 ] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        if verbose:
            print("--------")
            print(batch_generations[0])

        if not disable_tqdm:
            progress.update(len(batch_prompts) // num_return_sequences)

    assert len(generations) == len(
        prompts
    ) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations

@torch.no_grad()
def main(args):
    accelerator = Accelerator()
    set_random_seed(args.seed)

    print_rank_0("Loading model and tokenizer...")
    tokenizer = load_hf_tokenizer(args.tokenizer_path, max_seq_len=8192, fast_tokenizer=True)

    if '8b' in args.model_name_or_path.lower():
        tokenizer.unk_token_id =0
        # tokenizer.padding_side = "right"

    tokenizer.unk_token_id =0
    tokenizer.padding_side = "left"
    print_rank_0(f"tokenizer pad side: {tokenizer.padding_side}")

    ### create_hf_trained_model, use model local path
    # model = create_hf_trained_model(AutoModelForCausalLM,
    #                     args.model_name_or_path,
    #                     tokenizer,
    #                     ds_config=None,
    #                     dropout=args.dropout)

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config=None,
                            dropout=args.dropout)
    model = model.to(accelerator.device)
    args.dtype = torch.float16 if args.dtype == 'fp16' else torch.float32 if args.dtype == 'fp32' else torch.bfloat16
    model = model.to(args.dtype)
    model.eval()
    print_rank_0('model is dtype: {}'.format(model.dtype))


    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=4,  # Thorough search
        temperature=0.0,  # No randomness
        repetition_penalty=1.1,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        bos_token_id=model.config.bos_token_id,
    )


    for dataset in args.datasets:
        print_rank_0(f"Handling dataset: {dataset}")
        t_test_data = json.load(open(os.path.join(args.data_path, dataset, 'test.json'), 'r'))
        t_test_data = t_test_data[0:10]

        prompts = []
        for example in t_test_data:
            prompt = i_prompt.format_map(example)
            prompts.append(prompt)
        print_rank_0(prompts[0])


        accelerator.wait_for_everyone()
        device = accelerator.device
        with accelerator.split_between_processes(prompts) as prompt:
            model_outputs = []
            outputs = generate_completions(
                model=model,
                device=device,
                tokenizer=tokenizer,
                prompts=prompt,
                max_new_tokens=256,
                batch_size=args.per_device_eval_batch_size,
                stop_id_sequences=[[tokenizer.eos_token]],
                verbose=False,
                generation_config=generation_config)
            model_outputs.extend(outputs)
        outputs = gather_object(model_outputs)

        save_outputs = []
        correct = 0
        for example, output in zip(t_test_data, outputs):
            example['raw_output'] = output
            target = example["answer"].lower()
            predict = extract_answer(dataset, output)
            print("target", target, "predict", predict)
            if target == predict:
                correct += 1
            example['prediction'] = predict
            save_outputs.append(example)

        print_rank_0(f"Saving outputs to {args.output_dir}")

        weighted_acc = correct / len(t_test_data)
        print_rank_0("Dataset: {}, accuracy {:.1f}%, number of test data: {}".format(
            dataset, 
            weighted_acc * 100,
            len(t_test_data)
        ))
        
        dataset_output_dir = os.path.join(args.output_dir, dataset)
        os.makedirs(dataset_output_dir, exist_ok=True) 
        with open(os.path.join(dataset_output_dir, f"model_predictions.jsonl"),
                "w") as fout:
            for example in save_outputs:
                fout.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        type=str,
                        default="my_data/mmlu",
                        required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",  # Accepts 1 or more values
        default=["boolq"],
        help="List of dataset names"
    )
    parser.add_argument("--output_dir",
                        type=str,
                        default="eval/gsm",
                        required=True)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Path to tokenizer identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--dropout',
                        type=float,
                        default=0.0,
                        help='dropout rate of the model.')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16', 'fp32'],
                        help='Inference data type')
    parser.add_argument("--per_device_eval_batch_size",
                        type=int,
                        default=4,
                        help="batch size for evaluation.")
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help=
        "If given, the prompt will be encoded as a chat format with the roles in prompt."
    )
    args = parser.parse_args()

    #args.output_dir = os.path.join(args.output_dir, '-'.join(args.model.split('/')[-2:]))

    main(args)
