import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
import os
import torch
import time
import json
import itertools
import argparse

sys.path.append('./benchmarks/baseline')
from loadDataset import tokenize_text, DATASET_DICT


INPUT_TOKEN_LENGTH_PAD = None       # None: 不填充
INPUT_TOKEN_LENGTH_TRUNCATE = None  # None: 不截断

prompt_size_list = [512]
output_size_list = [16]
batch_size_list = [16]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # System config
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    parser.add_argument(
        "--n-token",
        type=int,
        default=32,
        help="Number of tokens to generate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation. Default is 1.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openai_humaneval",
        help="Dataset for generation. Default is None.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Path to model. Default is mistralai/Mixtral-8x7B-v0.1.",
    )
    parser.add_argument(
        "--disk-offload",
        action="store_true",
        default=False,
        help="Use disk offload for parameters.",
    )
    args = parser.parse_args()
    
    print(f"Batch size: {args.batch_size}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of tokens: {args.n_token}")
    print(f"Disk offload: {args.disk_offload}")
    print(f"Model: {args.model}")
    output_size = args.n_token
    batch_size = args.batch_size
    DATASET = args.dataset
    model_name = args.model

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    
    train_batch_size = 1 * world_size

    GB = 1 << 30
    if args.disk_offload:
        ds_config = {
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "nvme",
                    "pin_memory": True,
                    "nvme_path": "~/",
                    "buffer_count": 10,
                    "buffer_size": 1*GB,
                    "max_in_cpu": 5e9,
                },
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "steps_per_print": 2000,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False
        }
    else:
        ds_config = {
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True,
                },
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "steps_per_print": 2000,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False
        }


    # DeepSpeed Engine Launch
    print("\033[48;5;17mLaunching DeepSpeed engine\033[0m")

    dschf = HfDeepSpeedConfig(ds_config)

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()


    # Global tokenize
    print("\033[48;5;17mTokenizing input file\033[0m")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    global_records = []

    # Test cases
    for _, _, _ in itertools.product(prompt_size_list, output_size_list, batch_size_list):
        print(f"\033[48;5;17mCASE: DATASET {DATASET}, OUTPUT LENGTH {output_size}, BATCH SIZE {batch_size}\033[0m")
        
        GENERATION_BATCH_SIZE = batch_size
        MAX_NEW_TOKENS = output_size
        
        # Generate case-relavant tokens
        print("\033[48;5;17mGenerating input tokens\033[0m")
        
        # Tokenize text
        tokenization_config = {
            'token_length_pad': INPUT_TOKEN_LENGTH_PAD,
            'token_length_truncate': INPUT_TOKEN_LENGTH_TRUNCATE,
            'batch_size': GENERATION_BATCH_SIZE,
            **DATASET_DICT[DATASET],
        }
        input_tokens = tokenize_text(tokenizer, **tokenization_config)['input_ids']
        input_tokens = input_tokens.to(device=local_rank)

        # Inference
        print("\033[48;5;17mBegin inference\033[0m")
        with torch.no_grad():
            # Prefill
            print(f"\033[48;5;17mPREFILL\033[0m")
            start_time = time.time()
            outputs = ds_engine.module.generate(input_tokens, min_new_tokens=1, max_new_tokens=1)
            end_time = time.time()
            prefill_duration = end_time - start_time
            print(f"\033[48;5;17mTIME {prefill_duration:.3f}s\033[0m")
            text_out_prefill = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            
            
            # Decoding
            print(f"\033[48;5;17mDECODING\033[0m")
            start_time = time.time()
            outputs = ds_engine.module.generate(input_tokens, min_new_tokens=MAX_NEW_TOKENS, max_new_tokens=MAX_NEW_TOKENS)
            end_time = time.time()
            decoding_duration = end_time - start_time - prefill_duration
            print(f"\033[48;5;17mTIME {decoding_duration:.3f}s\033[0m")
            text_out_decoding = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            
            text_in = [tokenizer.decode(input_tokens_item, skip_special_tokens=True) for input_tokens_item in input_tokens]
            
            throughputs = batch_size * output_size / (prefill_duration + decoding_duration)
            print(f"\033[48;5;17mTHROUGHPUT: {throughputs} tokens/s \033[0m")
            
            # Record
            records = {
                "throughputs": throughputs,
                "prefill_duration": prefill_duration,
                "decoding_duration": decoding_duration,
                "text": [{
                    "prompt": prompt,
                    "prefill": prefill[len(prompt):],
                    "decoding": decoding[len(prompt):]
                } for prompt, prefill, decoding in zip(text_in, text_out_prefill, text_out_decoding)],
            }
            
            global_records.append({
                "dataset": DATASET,
                "output_length": output_size,
                "batch_size": batch_size,
                "input_token_length_truncate": INPUT_TOKEN_LENGTH_TRUNCATE,
                "input_token_length_pad": INPUT_TOKEN_LENGTH_PAD,
                **records
            })
            
            with open("log.txt", "a") as f:
                f.write(f"DATASET: {DATASET}\n")
                f.write(f"MODEL: {model_name}\n")
                f.write(f"Batch Size: {GENERATION_BATCH_SIZE}\n")
                f.write(f"Output Tokens: {output_size}\n")
                f.write(f"Disk Offload: {args.disk_offload}\n")
                f.write(f"Throughput: {throughputs} tokens/s\n")
                f.write("\n")
                
            print("\033[48;5;17mCase finished\033[0m")


    # Save
    # with open(f'result/result-{DATASET}-bs={batch_size}-output={output_size}.json', 'w') as f:
    #     json.dump(global_records, f, indent=4)

    torch.cuda.empty_cache()

    print("\033[48;5;17mAll finished\033[0m")