import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import time
import json
import os
from loadDataset import tokenize_text, DATASET_DICT
import argparse
import sys

model_path = "mistralai/Mixtral-8x7B-v0.1"

INPUT_TOKEN_LENGTH_PAD = None  # None: 不填充
INPUT_TOKEN_LENGTH_TRUNCATE = None  # None: 不截断
os.environ["TOKENIZERS_PARALLELISM"] = "false"
local_rank = "cuda:0"

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=128, help="batch size")
parser.add_argument("--n-token", type=int, default=16, help="output tokens")
parser.add_argument("--dataset", type=str, default='openai_humaneval', help="dataset name")
parser.add_argument("--model", type=str, default='mistralai/Mixtral-8x7B-v0.1', help="model path")
args = parser.parse_args()

batch_size = args.batch_size
output_size = args.n_token
DATASET = args.dataset
model_path = args.model
output_path = f"~/results_{output_size}_{batch_size}.txt"

torch.cuda.empty_cache()
# 初始化空模型权重，避免 OOM

with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

device_map = {
    'model.embed_tokens': local_rank, 'model.layers.0': local_rank, 'model.layers.1': 'cpu',
    'model.layers.2': 'cpu', 'model.layers.3': 'cpu', 'model.layers.4': 'cpu',
    'model.layers.5': 'cpu', 'model.layers.6': 'cpu', 'model.layers.7': 'cpu',
    'model.layers.8': 'cpu', 'model.layers.9': 'cpu', 'model.layers.10': 'cpu',
    'model.layers.11': 'cpu', 'model.layers.12': 'cpu', 'model.layers.13': 'cpu',
    'model.layers.14': 'cpu', 'model.layers.15': 'cpu', 'model.layers.16': 'cpu',
    'model.layers.17': 'cpu', 'model.layers.18': 'cpu', 'model.layers.19': 'cpu',
    'model.layers.20': 'cpu', 'model.layers.21': 'cpu', 'model.layers.22': 'cpu',
    'model.layers.23': 'cpu', 'model.layers.24': 'cpu', 'model.layers.25': 'cpu',
    'model.layers.26': 'cpu', 'model.layers.27': 'cpu', 'model.layers.28': 'cpu',
    'model.layers.29': 'cpu', 'model.layers.30': 'cpu', 'model.layers.31': 'cpu',
    'model.norm': 'cpu', 'model.rotary_emb': 'cpu', 'lm_head': 'cpu'
}

model = load_checkpoint_and_dispatch(
    model,
    model_path,
    device_map=device_map,
    max_memory={0: "24GiB", "cpu": "150GiB"},
    dtype=torch.bfloat16  # 使用 FP16 以减少显存占用
)

print(model.hf_device_map)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

input_text = "Hello my name is"
inputs = tokenizer(input_text, return_tensors="pt").to(local_rank)
print("warmup")
_ = model.generate(**inputs, max_new_tokens=1)
torch.cuda.synchronize(device=torch.device(local_rank))

tokenization_config = {
    'token_length_pad': INPUT_TOKEN_LENGTH_PAD,
    'token_length_truncate': INPUT_TOKEN_LENGTH_TRUNCATE,
    'batch_size': 4,
    **DATASET_DICT[DATASET],
}

total_len, prompts = tokenize_text(tokenizer, **tokenization_config)

total_tokens = 0  # 记录总 tokens 数量
total_time = 0
# 推理

sum = 1
with open(output_path, "w", encoding="utf-8") as f_out:
    for batch_idx in range(sum):
        try:
            print(f"推理 batch {batch_idx + 1} / {sum}")
            tokenization_config = {
                'token_length_pad': INPUT_TOKEN_LENGTH_PAD,
                'token_length_truncate': INPUT_TOKEN_LENGTH_TRUNCATE,
                'batch_size': batch_size,
                'skip': batch_idx * batch_size,
                **DATASET_DICT[DATASET],
            }
            _, inputs = tokenize_text(tokenizer, **tokenization_config)
            inputs = inputs['input_ids']
            inputs = inputs.to(device=local_rank)
            start = time.time()
            outputs = model.generate(inputs, max_new_tokens=output_size)
            end = time.time()
            # output_tokens = sum([len(output) for output in outputs])
            total_tokens += len(inputs) * output_size
            total_time += end - start
            for i, output in enumerate(outputs):
                decoded = tokenizer.decode(output, skip_special_tokens=True, errors='ignore')
                f_out.write(f"[Prompt {batch_idx * batch_size + i + 1}]: \n")
                f_out.write(f"[Output]: {decoded}\n")
                f_out.write(f"[Time]: {end - start:.2f}s\n\n")
                print(f"{batch_idx * batch_size + i + 1}: {decoded}")

            # 释放资源
            del inputs, outputs
            torch.cuda.empty_cache()


        except Exception as e:
            print(f"Batch {batch_idx + 1} 失败: {e}")
            f_out.write(f"Batch {batch_idx + 1} 失败: {e}")
            sys.exit(1)
            # for i, prompt in enumerate(batch_prompts):
            #    f_out.write(f"[Prompt {batch_idx * batch_size + i + 1}]: {prompt}\n[Error]: {str(e)}\n\n")
    # total_end = time.time()

    tokens_per_second = total_tokens / total_time if total_time > 0 else 0

    f_out.write(f"\n总推理时间: {total_time:.3f} 秒\n")
    f_out.write(f"总 tokens 数量: {total_tokens}\n")
    f_out.write(f"吞吐量: {tokens_per_second:.3f} tokens/s\n")
print(f"总推理时间: {total_time:.3f} 秒")
print(f"总 tokens 数量: {total_tokens}")
print(f"吞吐量: {tokens_per_second:.3f} tokens/s")
print(f"所有推理完成，结果已保存到：{output_path}")
print(f"所有推理完成，结果已保存到：{output_path}")