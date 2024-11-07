import os
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F  # Add this import
import math
#from datasets import load_dataset
# Set custom cache directory
import os
# custom_cache_dir = "/lustre/vescratch1/ceodspspectrum/cache_hf/

custom_cache_dir = "/home/user/Desktop/LLMs/TTLoRAMoE/Running_TTLoRA/cache_hf/"

#custom_cache_dir = '/lustre/scratch5/ceodspspectrum/tmp_ray/'
os.environ['HF_DATASETS_CACHE'] = custom_cache_dir
#set_caching_dir(custom_cache_dir)


def load_dataset_(data_name):
    path = '/home/user/Desktop/LLMs/TTLoRAMoE/Running_TTLoRA'+"/data"
    data_path = os.path.join(path, data_name)
    dataset = load_dataset(data_path)
    # dataset = load_dataset("glue", data_name)
    return dataset
def get_tokenizer(model_path, data_name, dataset):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

    print("Tokenizer input max length:", tokenizer.model_max_length)
    print("Tokenizer vocabulary size:", tokenizer.vocab_size)

    ### set the special tokens needed during tokenization
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_text(batch):
        if data_name == "sst2":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "mrpc":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "cola":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "qnli":
            return tokenizer(batch["question"], batch['sentence'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "rte":
            print("got rte")
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "mnli":
            return tokenizer(batch["premise"], batch['hypothesis'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "qqp":
            return tokenizer(batch["question1"], batch['question2'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "stsb":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding=True)
    ### map the words in the dataset to the token values of the loaded tokenizer
    tokenized = dataset.map(tokenize_text, batched=True, batch_size=None)
    ### which columns to keep?
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return tokenized
