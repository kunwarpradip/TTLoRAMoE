import os, sys
from datasets import load_dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F  # Add this import
import math
import tensorly as tl
from tensorly.decomposition import tensor_train
from transformers import AutoTokenizer

import pickle

from argparse import ArgumentParser

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils import get_tokenizer, load_dataset_
from ttlora_wrapper import LoRATTLinearWrapper, get_tensor_shape, get_tt_rank
from transformers import AutoModelForSequenceClassification
from model import CustomLightningModule
from functools import partial


def load_dataset_(data_name):
    path = '/home/user/Desktop/LLMs/TTLoRAMoE/Running_TTLoRA/running_ttlora_codes' + "/data"
    data_path = os.path.join(path, data_name)
    dataset = load_dataset(data_path)
    return dataset

def get_tokenizer(model_path, data_name, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Tokenizer input max length:", tokenizer.model_max_length)
    print("Tokenizer vocabulary size:", tokenizer.vocab_size)

    # Set the special tokens needed during tokenization
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

    # Select the first two examples from the dataset for demonstration
    small_dataset = dataset

    # Tokenize the small dataset
    tokenized = small_dataset.map(tokenize_text, batched=True, batch_size=None)
    
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Print the tokenized output for the first two examples
    # for i in range(2):
    #     print(f"Example {i+1}:")
    #     print("Original Text:", small_dataset['train'][i]['sentence'])
    #     print(((tokenized['train'][i])))
    #     print("Token IDs:", tokenized['train'][i]['input_ids'])
    #     tokens = tokenizer.convert_ids_to_tokens(tokenized['train'][i]['input_ids'])
    #     print("Tokens:", tokens)
    # #     print()

    # # Map the entire dataset
    # tokenized = dataset.map(tokenize_text, batched=True, batch_size=None)
    

    return tokenized

def count_and_print_parameters(model):
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        print(f"Name: {name}")
        print(f"Shape: {param.shape}")
        print(f"Requires Grad: {param.requires_grad}")
        print(f"Values: {param.data[1:1]}")  # Print the first 5 values for brevity
        print()
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Total number of parameters: {total_params}")
    print(f"Total number of trainable parameters: {trainable_params}")

# Call the function

from transformers import AutoModelForSequenceClassification
import torch

if __name__ == "__main__":
    
    parser = ArgumentParser()
    # parser.add_argument("--ranks", type=int, default=5)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--data", type=str, default = 'cola')
    # parser.add_argument("--shape", type=int, default=7)
    # parser.add_argument("--bs", type=int, default=128)
    args = parser.parse_args()
    if args.data == "mnli":
        model = AutoModelForSequenceClassification.from_pretrained(
            "/home/user/Desktop/LLMs/TTLoRAMoE/Running_TTLoRA/checkpoints/roberta", num_labels=3)
    if args.data == "stsb":
        model = AutoModelForSequenceClassification.from_pretrained(
            "/home/user/Desktop/LLMs/TTLoRAMoE/Running_TTLoRA/checkpoints/roberta", num_labels=1)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            "/home/user/Desktop/LLMs/TTLoRAMoE/Running_TTLoRA/checkpoints/roberta", num_labels=2)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "microsoft/deberta-base", num_labels=2) ##########

    model.config.pad_token_id = model.config.eos_token_id
    # print(model)
    for param in model.parameters():
        param.requires_grad = False

    config = {
        "shapes": [12, 8, 8, 2, 2, 2, 8, 12], 
        "ranks": 2,
        "alpha": 8,
        "learning_rate":  1e-3,
    }

    tt_shape_768_768 = get_tensor_shape(config["shapes"])
    tt_rank = get_tt_rank(config["ranks"], tt_shape_768_768)


    loratt_alpha=config["alpha"]
    loretta_dropout = 0.05
    loratt_query = True
    loretta_value = True

    layers = []

    # assign_lora = partial(LoRALinearWrapper, rank=lora_r, alpha=lora_alpha)

    assign_loretta = partial(LoRATTLinearWrapper, tt_shape = tt_shape_768_768, tt_rank=tt_rank, alpha=loratt_alpha)
    
    i=0
    for layer in model.roberta.encoder.layer:
        i+=1
        if loratt_query:
            layer.attention.self.query = assign_loretta(layer.attention.self.query, 0)
        if loretta_value:
            layer.attention.self.value = assign_loretta(layer.attention.self.value, 2)
    
    # print(model)
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # print("Total number of trainable parameters:", count_parameters(model))
    