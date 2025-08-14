import sys
import torch
from torch.nn.functional import pad
sys.path.append('../src')
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, BatchEncoding
from transformers.generation import AssistedCandidateGenerator, GenerationConfig
from transformers.generation.parallelism import MultiThreadGenerationInterface, AssistantProcessInterface
from transformers.generation.logger import LogConfig
from datasets import load_dataset
import argparse

import threading
from threading import Thread
import torch.multiprocessing as mp

from typing import Optional, Any, List, Callable
from copy import copy
import itertools

DATASET_DICT = {
    'openai_humaneval': {
        'dataset_path': './benchmarks/openai_humaneval/dataset/openai_humaneval/test-00000-of-00001.parquet',
        'dataset_type': 'parquet',
        'dataset_extractor': lambda x: x['train']['prompt'],
    },
    'summeval': {
        'dataset_path': './benchmarks/summeval_data/dataset/data/test-00000-of-00001-35901af5f6649399.parquet',
        'dataset_type': 'parquet',
        'dataset_extractor': lambda x: x['train']['text'],
        'dataset_preprocessor': lambda x: 'Article: ' + x + ' Summary: ',
    },
    'samsum': {
        'dataset_path': './benchmarks/samsum/dataset/samsum/train/0000.parquet',
        'dataset_type': 'parquet',
        'dataset_extractor': lambda x: x['train']['dialogue'],
        'dataset_preprocessor': lambda x: 'Dialogue: ' + x + ' Summary: ',
    },
    'ceval_exam': {
        'dataset_path': './benchmarks/ceval-exam/dataset/civil_servant/test-00000-of-00001.parquet',
        'dataset_type': 'parquet',
        'dataset_extractor': lambda x: [y['question'] + ' A.' + y['A'] + ' B.' + y['B'] + ' C.' + y['C'] + ' D.' + y['D'] for y in x['train']],
        'dataset_preprocessor': lambda x: 'Question: ' + x + ' Answer and explanation: ',
    },
}

def load_model(
    model_path: str,
    device: str = 'auto',
    gpu_device: str = "cuda:0",
    is_target_model: bool = False,
) -> PreTrainedModel:
    # 加载模型
    if is_target_model:
        torch.cuda.set_device(device=gpu_device)
        
        # 1. 先加载模型到CPU
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        
        # 2. 将一半的参数固定到内存中
        params = list(model.parameters())
        half = len(params) // 2
        print(f"Pinning memory for {half} parameters...")
        import time
        start_time = time.time()
        for i, param in enumerate(params):
            if i < half:
                if param.is_cuda:
                    param.data = param.data.cpu()  # 确保在CPU上
                param.data = param.data.pin_memory()  # 固定内存
        end_time = time.time()
        print(f"Pinning memory time: {end_time - start_time:.2f}s")

        # 3. 将模型移到目标设备
        if device != 'auto':
            model.to(device)
    else:
        print(f"Loading model to {device}...")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        if device != 'auto':
            model = model.to(device)
        
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    return model

def load_tokenizer(
    tokenizer_path: str,
) -> PreTrainedTokenizer:
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, torch_dtype=torch.bfloat16)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def tokenize_text(
    tokenizer: PreTrainedTokenizer,
    dataset_path: str = None,
    dataset_type: str = None,
    dataset_extractor: Callable[[Any], List[str]] = None,
    dataset_preprocessor: Optional[Callable[[List[str]], List[str]]] = None,
    text_path: str = './examples/romeo-and-juliet.txt',
    device: str = 'auto',
    token_length_pad: Optional[int] = None,
    token_length_truncate: Optional[int] = None,
    batch_size: int = 1,
    repeat: bool = False,
    skip: int = 0,
) -> BatchEncoding:
    if dataset_path is not None:
        dataset = load_dataset(dataset_type, data_files=dataset_path)
        # print(dataset)
        dataset = dataset_extractor(dataset) # 提取文本列
        
        if skip + batch_size <= len(dataset): 
            dataset = dataset[skip:skip+batch_size] # 跳过前skip行，获取batch_size行
        else:
            dataset = dataset[-skip-batch_size:]    # 获取全部行
        if dataset_preprocessor is not None:
            for i in range(len(dataset)):
                dataset[i] = dataset_preprocessor(dataset[i]) # 逐行处理数据
                # print(dataset[i])
        
        if token_length_truncate is not None:
            tokenization_params = {'truncation': True, 'max_length': token_length_truncate}
        else:
            tokenization_params = {}
            
        inputs = tokenizer(
            dataset,
            padding=True, # 自动填充至批次内最长序列
            return_tensors='pt', # 返回 PyTorch 张量
            **tokenization_params,) # 截断参数（max_length=token_length_truncate）
        
        # Adjust batch size
        if repeat: # 重复首个样本填充批次
            inputs['input_ids'] = inputs['input_ids'][:1]
            inputs['attention_mask'] = inputs['attention_mask'][:1]
        while inputs['input_ids'].shape[0] < batch_size: # 复制样本直至满足批次大小
            inputs['input_ids'] = torch.cat([inputs['input_ids'], inputs['input_ids']])
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], inputs['attention_mask']])
        if inputs['input_ids'].shape[0] > batch_size: # 截断多余样本
            inputs['input_ids'] = inputs['input_ids'][:batch_size]
            inputs['attention_mask'] = inputs['attention_mask'][:batch_size]
            
        # Adjust token length
        if token_length_pad is not None and inputs['input_ids'].shape[1] < token_length_pad:
            pad_size = token_length_pad - inputs['input_ids'].shape[1] # 需要填充的字符长度
            inputs['input_ids'] = pad(inputs['input_ids'], (pad_size, 0), value=tokenizer.pad_token_id) # 填充字符（pad_size, 0）表示对最后一个维度左边扩充pad_size列，右边不扩充
            inputs['attention_mask'] = pad(inputs['attention_mask'], (pad_size, 0), value=0) # 同步填充注意力掩码，标识填充部分无需模型关注
        
        if device != 'auto': # 显式指定设备（GPU/CPU）
            inputs.to(device)
        
    else:
        # 读取文本内容
        text = ""
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 将文本分词
        tokens = tokenizer.tokenize(text) # 分词（如 ["Let", "'", "s", ...]）

        # 分割文本为 batch
        batch = []
        for i in range(0, len(tokens), token_length_truncate):
            chunk = tokens[i:i + token_length_truncate]
            batch.append(chunk)
            if len(batch) == batch_size:
                break

        # 对输入batch进行tokenize
        inputs = tokenizer(
            batch, 
            is_split_into_words=True,          # 因为输入是 token 列表，已经分词
            padding=True,                      # 添加 padding 自动填充至批次内最长序列长度
            truncation=True,                   # 确保长度限制在模型支持的最大 token 长度
            max_length=token_length_truncate,  # 设置最大长度为 token_length
            return_tensors='pt'                # 返回 PyTorch tensors
        )
        if device != 'auto':
            inputs.to(device)
    
    return inputs

MAIN_MODEL_PATH = "mistralai/Mixtral-8x7B-v0.1"
ASSISTANT_MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET = "openai_humaneval"

SPECULATIVE_DECODING = True
PARALLEL_ASSISTANT = True
PARALLEL_GENERATION = True

INPUT_TOKEN_LENGTH_PAD = None       # None: 不填充
INPUT_TOKEN_LENGTH_TRUNCATE = None  # None: 不截断
MAX_NEW_TOKENS = 16
GENERATION_BATCH_SIZE = 16
PREFILL_BATCH_SIZE = 2             # None: 设置为 GENERATION_BATCH_SIZE
ASSISTANT_BATCH_SIZE = 5           # None: 设置为 GENERATION_BATCH_SIZE
ASSISTANT_MAX_NEW_TOKENS = 5

LOG_CONFIG = LogConfig(
    timer=True,
    generation_process=True,
    generation_statistics=True,
    # n_show_output=0,
    n_show_output=4,
)

@torch.no_grad()
def assistant_model_generation(
    assistant_process_interface: AssistantProcessInterface,
    assistant_batch_size: Optional[int] = None,
    assistant_max_new_tokens: Optional[int] = None,
    ASSISTANT_MODEL_PATH: Optional[str] = "mistralai/Mistral-7B-Instruct-v0.2",
    gpu_device: Optional[str] = 'cuda:0',
) -> None:
    print(f"assistant model generation gpu device: {gpu_device}")
    assistant_model = load_model(ASSISTANT_MODEL_PATH, device='cpu') # 将辅助模型加载进CPU
    if assistant_max_new_tokens is not None: # 不启用动态推测解码
        assistant_model.generation_config.assistant_confidence_threshold = 0.0
        assistant_model.generation_config.num_assistant_tokens = assistant_max_new_tokens
    model_on_gpu = False
    
    assistant_process_interface.assistant_ready_event.set() # 设置辅助模型空闲
    
    while True:
        # Wait for input
        assistant_process_interface.input_ready_event.wait() # 辅助模型等待接收输入
        if not model_on_gpu:
            # Wait until all batches finish prefilling
            for event in assistant_process_interface.batch_finish_prefilling_events: # 等待两个批次的输入完成prefill
                event.wait()
            
            assistant_model.to(gpu_device) # 加载辅助模型进GPU
            candidate_generator = AssistedCandidateGenerator(
                input_ids=torch.Tensor(),
                assistant_model=assistant_model,
                generation_config=GenerationConfig(
                    max_length=10**18,
                ),
                model_kwargs={
                    'num_logits_to_keep': 1,
                    'use_cache': True,
                },
            )
            model_on_gpu = True
        
        # Receive input
        input_ids = assistant_process_interface.get_input_ids().to(gpu_device)
        attention_mask = assistant_process_interface.get_attention_mask().to(gpu_device)
        assistant_process_interface.input_ready_event.clear()
        
        # Split input into chunks
        if assistant_batch_size is None:
            assistant_batch_size = input_ids.shape[0]
        input_ids_list = list(input_ids.split(assistant_batch_size))
        attention_mask_list = list(attention_mask.split(assistant_batch_size))
        
        # Candidate generation
        candidate_input_ids_list = []
        candidate_logits_list = []
        for input_ids_chunk, attention_mask_chunk in zip(input_ids_list, attention_mask_list):
            candidate_input_ids_chunk, candidate_logits_chunk = candidate_generator.get_candidates(input_ids_chunk, attention_mask_chunk)
            candidate_input_ids_list.append(candidate_input_ids_chunk)
            candidate_logits_list.append(candidate_logits_chunk)
        candidate_input_ids_list = list(candidate_input_ids_list)
        candidate_logits_list = list(candidate_logits_list)
        
        # Padding
        candidate_token_length_list = [candidate_input_ids_chunk.shape[1] for candidate_input_ids_chunk in candidate_input_ids_list]
        candidate_token_length = max(candidate_token_length_list)
        
        for i in range(len(candidate_input_ids_list)):
            pad_size = candidate_token_length - candidate_token_length_list[i]
            if pad_size > 0:
                candidate_input_ids_list[i] = pad(candidate_input_ids_list[i], (0, pad_size), value=0)
                candidate_logits_list[i] = pad(candidate_logits_list[i], (0, 0, 0, pad_size), value=0.0)
        
        # Concatenation
        candidate_input_ids = torch.cat(candidate_input_ids_list)
        candidate_logits = torch.cat(candidate_logits_list)
        
        # Transmit output
        assistant_process_interface.set_input_ids(candidate_input_ids)
        assistant_process_interface.set_candidate_logits(candidate_logits)
        
        assistant_process_interface.output_ready_event.set()
        
        torch.cuda.empty_cache()


THREAD_GENERATION_RESULT = []
def thread_generation(model_generate, *args, **kwargs):
    result = model_generate(*args, **kwargs)
    THREAD_GENERATION_RESULT.append(result)

# python examples/run.py --input-token-length-truncate 128 --cuda 0
if __name__ == '__main__':
    # Common generation configuration
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-token-length-pad",
        type=int,
        default=None,
        help="Input token length pad. Default is None.",
    )
    parser.add_argument(
        "--input-token-length-truncate",
        type=int,
        default=None,
        help="Input token length truncate. Default is None.",
    )
    parser.add_argument(
        "--n-token",
        type=int,
        default=16,
        help="Number of tokens to generate.",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=32,
        help="Batch size for generation. Default is 32.",
    )
    parser.add_argument(
        "--prefill-batch-size",
        type=int,
        default=16,
        help="Batch size for prefill. Default is 16.",
    )
    parser.add_argument(
        "--assisant-batch-size",
        type=int,
        default=16,
        help="Batch size for assistant. Default is 16.",
    )
    parser.add_argument(
        "--assisant-max-new-tokens",
        type=int,
        default=5,
        help="Max new tokens for assistant. Default is 5.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset for generation. Default is None.",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="/home/yyf/data/Mixtral-8x7B-v0.1",
        help="Path to target model. Default is /home/yyf/data/Mixtral-8x7B-v0.1.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default="/home/yyf/data/Mistral-7B-Instruct-v0.2",
        help="Path to draft model. Default is /home/yyf/data/Mistral-7B-Instruct-v0.2.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="CUDA device ID. Default is 0.",
    )
    args = parser.parse_args()
    
    INPUT_TOKEN_LENGTH_PAD = args.input_token_length_pad
    INPUT_TOKEN_LENGTH_TRUNCATE = args.input_token_length_truncate
    
    MAX_NEW_TOKENS = args.n_token
    GENERATION_BATCH_SIZE = args.generation_batch_size
    PREFILL_BATCH_SIZE = args.prefill_batch_size
    ASSISTANT_BATCH_SIZE = args.assisant_batch_size
    ASSISTANT_MAX_NEW_TOKENS = args.assisant_max_new_tokens
    
    DATASET = args.dataset
    MAIN_MODEL_PATH = args.target_model
    ASSISTANT_MODEL_PATH = args.draft_model
    GPU_DEVICE = f'cuda:{args.cuda}'
    
    print(f"INPUT_TOKEN_LENGTH_PAD: {INPUT_TOKEN_LENGTH_PAD}")
    print(f"INPUT_TOKEN_LENGTH_TRUNCATE: {INPUT_TOKEN_LENGTH_TRUNCATE}")
    print(f"MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
    print(f"GENERATION_BATCH_SIZE: {GENERATION_BATCH_SIZE}")
    print(f"PREFILL_BATCH_SIZE: {PREFILL_BATCH_SIZE}")
    print(f"ASSISTANT_BATCH_SIZE: {ASSISTANT_BATCH_SIZE}")
    print(f"ASSISTANT_MAX_NEW_TOKENS: {ASSISTANT_MAX_NEW_TOKENS}")
    print(f"DATASET: {DATASET}")
    print(f"TARGET MODEL: {MAIN_MODEL_PATH}")
    print(f"DRAFT  MODEL: {ASSISTANT_MODEL_PATH}")
    print(f"CUDA DEVICE: {GPU_DEVICE}")
    
    num_of_new_tokens = [0, 0] # 双批次并行推理，一个batch做推测，另一个batch做验证，此列表表示两个batch各自的已生成tokens数量
    generation_kwargs = {
        'max_new_tokens': MAX_NEW_TOKENS, # 拟生成token数量
        'prefill_batch_size': PREFILL_BATCH_SIZE, # prefill的batch_size
        'log_config': LOG_CONFIG, # 计时器
        'num_of_new_tokens': num_of_new_tokens, # 双批次已生成token数量
        'gpu_device': GPU_DEVICE,
    }
    if PARALLEL_GENERATION:
        generation_kwargs_1 = copy(generation_kwargs)

    # Tokenize text
    tokenization_config = {
        'token_length_pad': INPUT_TOKEN_LENGTH_PAD,
        'token_length_truncate': INPUT_TOKEN_LENGTH_TRUNCATE,
        'batch_size': GENERATION_BATCH_SIZE,
        **DATASET_DICT[DATASET],
    } if DATASET is not None else {
        'token_length_pad': INPUT_TOKEN_LENGTH_PAD,
        'token_length_truncate': INPUT_TOKEN_LENGTH_TRUNCATE,
        'batch_size': GENERATION_BATCH_SIZE,
    }

    tokenizer = load_tokenizer(MAIN_MODEL_PATH) # 加载tokenizer
    # inputs['input_ids'].shape = (batch_size, token_length_truncate) = inputs['attention_mask'].shape
    inputs = tokenize_text(tokenizer, **tokenization_config) # inputs是一个字典, 包含input_ids和attention_mask两个键值对
    # 在generation_kwargs里添加input_ids和attention_mask
    generation_kwargs.update(inputs)  # 此时inputs['input_ids'].device = inputs['attention_mask'].device = CPU
    
    vocab_size = tokenizer.vocab_size # 32000
    input_token_length = inputs['input_ids'].shape[1] # 128

    if SPECULATIVE_DECODING and PARALLEL_GENERATION:
        inputs_1 = tokenize_text(tokenizer, skip=GENERATION_BATCH_SIZE, **tokenization_config)
        generation_kwargs_1.update(inputs_1) # 此时inputs_1['input_ids'].device = inputs_1['attention_mask'].device = CPU
    
    # Prepare assistant model configuration
    if SPECULATIVE_DECODING:
        if PARALLEL_ASSISTANT:
            number_of_threads = 2 if PARALLEL_GENERATION else 1
            
            assistant_process_interface = AssistantProcessInterface(
                vocab_size=vocab_size, # 词汇表长度
                input_token_length=input_token_length, # prompt长度
                max_new_tokens=MAX_NEW_TOKENS, # 拟生成token数量
                generation_batch_size=GENERATION_BATCH_SIZE, # 主模型推理batch_size
                assistant_batch_size=ASSISTANT_BATCH_SIZE, # 辅助模型推理batch_size
                assistant_max_new_tokens=ASSISTANT_MAX_NEW_TOKENS, # 辅助模型一次推测token的数量
                number_of_threads=number_of_threads,
            )
            
            # Start assistant process
            assistant_process = mp.Process(
                target=assistant_model_generation,
                args=(assistant_process_interface,),
                kwargs={
                    'assistant_batch_size': ASSISTANT_BATCH_SIZE,
                    'assistant_max_new_tokens': ASSISTANT_MAX_NEW_TOKENS,
                    'ASSISTANT_MODEL_PATH': ASSISTANT_MODEL_PATH,
                    'gpu_device': GPU_DEVICE,
                },
                daemon=True, # 设置为守护进程，主进程终止时自动终止子进程，避免资源泄漏
            )
            assistant_process.start()
            
            generation_kwargs['assistant_process_interface'] = assistant_process_interface
            if PARALLEL_GENERATION:
                generation_kwargs_1['assistant_process_interface'] = assistant_process_interface
        else:
            # Load assistant model
            assistant_model = load_model(ASSISTANT_MODEL_PATH, device=GPU_DEVICE) # 将draft model加载进GPU
            assistant_model.generation_config.assistant_confidence_threshold = 0.0 # 关闭动态推测解码
            
            generation_kwargs['assistant_model'] = assistant_model
            if PARALLEL_GENERATION:
                generation_kwargs_1['assistant_model'] = assistant_model
            
    # Load main model
    model = load_model(MAIN_MODEL_PATH, device='cpu', gpu_device=GPU_DEVICE, is_target_model=True) # 将target model加载进CPU
    model.init_gpu_device(GPU_DEVICE) # 在GPU上预分配一层的两个激活专家的空间
        
    # Prepare multi-thread generation
    if SPECULATIVE_DECODING and PARALLEL_GENERATION:
        # Prepare synchronization events
        batch_finish_events = [threading.Event(), threading.Event()]
        batch_finish_events[1].set()
        alive_flags = [True, True]

        batch_generation_interfaces_0 = MultiThreadGenerationInterface(
            thread_id=0,
            batch_finish_events=batch_finish_events,
            alive_flags=alive_flags,
        )
        batch_generation_interfaces_1 = MultiThreadGenerationInterface(
            thread_id=1,
            batch_finish_events=batch_finish_events,
            alive_flags=alive_flags,
        )
        
        generation_kwargs['multi_thread_interface'] = batch_generation_interfaces_0
        generation_kwargs_1['multi_thread_interface'] = batch_generation_interfaces_1


    # Start generation
    with torch.no_grad():
        LOG_CONFIG.start_timer()
        
        # Multi-thread generation
        if PARALLEL_GENERATION:
            batch_thread = Thread(target=thread_generation, args=(model.generate,), kwargs=generation_kwargs_1)
            batch_thread.start()
        
        # Generation
        outputs = model.generate(**generation_kwargs)
        
        # Wait for multi-thread generation
        if PARALLEL_GENERATION:
            batch_thread.join()
            outputs_1 = THREAD_GENERATION_RESULT.pop()
            
    # print to log file
    with open('log.txt', 'a') as f:
        f.write("--------------- REPORT ---------------\n")
        f.write("RUN INFO:\n")
        f.write(f"  TARGET MODEL: {MAIN_MODEL_PATH}\n")
        f.write(f"  DRAFT  MODEL: {ASSISTANT_MODEL_PATH}\n")
        f.write(f"  DATASET: {DATASET}\n")
        f.write(f"  GPU DEVICE: {GPU_DEVICE}\n")
        f.write(f"  SPECULATIVE_DECODING: {SPECULATIVE_DECODING}\n")
        f.write(f"  PARALLEL_ASSISTANT: {PARALLEL_ASSISTANT}\n")
        f.write(f"  PARALLEL_GENERATION: {PARALLEL_GENERATION}\n")
        f.write(f"  INPUT_TOKEN_LENGTH: {input_token_length}\n")
        f.write(f"  MAX_NEW_TOKENS: {MAX_NEW_TOKENS}\n")
        f.write(f"  GENERATION_BATCH_SIZE: {GENERATION_BATCH_SIZE}{' x 2' if PARALLEL_GENERATION else ''}\n")
        f.write(f"  PREFILL_BATCH_SIZE: {PREFILL_BATCH_SIZE}\n")
        f.write(f"  ASSISTANT_BATCH_SIZE: {ASSISTANT_BATCH_SIZE}\n")
        f.write(f"  ASSISTANT_MAX_NEW_TOKENS: {ASSISTANT_MAX_NEW_TOKENS}\n")
        
        f.write("STATISTICS:\n")
        
        total_tokens = sum(num_of_new_tokens)
        f.write(f"  Total tokens generated: {total_tokens}\n")
        if PARALLEL_GENERATION:
            f.write(f"      where {num_of_new_tokens[0]} generated in batch 0, {num_of_new_tokens[1]} generated in batch 1\n")
            
        total_time = LOG_CONFIG.get_time()
        f.write(f"  Total time elapsed: {total_time:.2f}s\n")
        
        throughput = total_tokens / total_time
        f.write(f"  Throughput: {throughput:.3f} tokens/s\n")
        
        f.write("\n")
        
        f.write("OUTPUT:\n")
        if PARALLEL_GENERATION:
            f.write(f">> BATCH 0\n\n")
            
        for line in outputs[:LOG_CONFIG.n_show_output]:
            f.write(
                tokenizer.decode(line[:input_token_length], skip_special_tokens=True) +
                "\033[32m" +
                tokenizer.decode(line[input_token_length:]) +
                "\033[0m" +
                "\n\n"
            )
            
        if PARALLEL_GENERATION:
            f.write(f">> BATCH 1\n\n")
            
            for line in outputs_1[:LOG_CONFIG.n_show_output]:
                f.write(
                    tokenizer.decode(line[:input_token_length], skip_special_tokens=True) +
                    "\033[32m" +
                    tokenizer.decode(line[input_token_length:]) +
                    "\033[0m" +
                    "\n\n"
                )
                
        count = 0
        f.write(">> Early stop\n\n")
        for line in outputs:
            if len(line) - input_token_length < MAX_NEW_TOKENS:
                f.write(
                    tokenizer.decode(line[:input_token_length], skip_special_tokens=True) +
                    "\033[32m" +
                    tokenizer.decode(line[input_token_length:], skip_special_tokens=True) +
                    "\033[0m" +
                    "\n\n"
                )
                count += 1
                if count >= LOG_CONFIG.n_show_output:
                    break
