import numpy as np
import torch
from torch.nn.functional import pad
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, BatchEncoding

from typing import Optional, Any, List, Callable

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

def tokenize_text(
    tokenizer: PreTrainedTokenizer,
    dataset_path: str,
    dataset_type: str,
    dataset_extractor: Callable[[Any], List[str]],
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
        print(dataset)
        dataset = dataset_extractor(dataset)
        print(f"total: {len(dataset)}")
        datalen = np.array([len(data) for data in dataset])
        print(f"max: {datalen.max()}")
        print(f"avg: {datalen.mean()}")
        if skip + batch_size <= len(dataset):
            dataset = dataset[skip:skip+batch_size]
        else:
            dataset = dataset[-skip-batch_size:]
        if dataset_preprocessor is not None:
            for i in range(len(dataset)):
                dataset[i] = dataset_preprocessor(dataset[i])
                # print(f"length: {len(dataset[i])}")
        
        if token_length_truncate is not None:
            tokenization_params = {'truncation': True, 'max_length': token_length_truncate}
        else:
            tokenization_params = {}
            
        inputs = tokenizer(
            dataset,
            padding=True,
            return_tensors='pt',
            **tokenization_params,)
        
        # Adjust batch size
        if repeat:
            inputs['input_ids'] = inputs['input_ids'][:1]
            inputs['attention_mask'] = inputs['attention_mask'][:1]
        while inputs['input_ids'].shape[0] < batch_size:
            inputs['input_ids'] = torch.cat([inputs['input_ids'], inputs['input_ids']])
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], inputs['attention_mask']])
        if inputs['input_ids'].shape[0] > batch_size:
            inputs['input_ids'] = inputs['input_ids'][:batch_size]
            inputs['attention_mask'] = inputs['attention_mask'][:batch_size]
            
        # Adjust token length
        if token_length_pad is not None and inputs['input_ids'].shape[1] < token_length_pad:
            pad_size = token_length_pad - inputs['input_ids'].shape[1]
            inputs['input_ids'] = pad(inputs['input_ids'], (pad_size, 0), value=tokenizer.pad_token_id)
            inputs['attention_mask'] = pad(inputs['attention_mask'], (pad_size, 0), value=0)
        
        if device != 'auto':
            inputs.to(device)
        
    else:
        # 读取文本内容
        text = ""
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 将文本分词
        tokens = tokenizer.tokenize(text)

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
            is_split_into_words=True,          # 因为输入是 token 列表
            padding=True,                      # 添加 padding 使每个输入等长
            truncation=True,                   # 确保长度限制在模型支持的最大 token 长度
            max_length=token_length_truncate,  # 设置最大长度为 token_length
            return_tensors='pt'                # 返回 PyTorch tensors
        )
        if device != 'auto':
            inputs.to(device)
    
    return inputs