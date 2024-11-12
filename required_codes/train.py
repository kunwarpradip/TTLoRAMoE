### import the required libraries

import os
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


from utils import get_tokenizer, load_dataset_
from ttlora_wrapper import LoRATTLinearWrapper, get_tensor_shape, get_tt_rank
from transformers import AutoModelForSequenceClassification
from model import CustomLightningModule
from functools import partial
# Ensure that the environment is properly configured for Ray
# (This is optional and environment-specific)

def train_without_ray(config):

    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this notebook.")

    ### load the dataset using args.data (defaulr: cola) argument from cli 
    dataset = load_dataset_(args.data)
    # print(dataset)

    tokenized = get_tokenizer("../checkpoints/roberta", args.data, dataset)



    train_dataset = tokenized["train"]
    # print(tokenized.keys())

    #question: don't we enter mnli only as argument?
    if args.data == "mnli--":
        val_dataset = tokenized["validation_matched"]
    else:
        val_dataset = tokenized["validation"]



    ### create train, validation and test dataloader that will be used during training, testing and validation. 
    # The dataloader specifies the number of rows in each batch and how many gpus to use
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True,   #data shuffles at the beginning of each epoch
        num_workers=4   #separate subprocesses to load data in parallel
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=128,
        num_workers=4
        #no need to shuffle the validation data as to get the consistent evaluations
    )



    if args.data == "mnli":
        model = AutoModelForSequenceClassification.from_pretrained(
            "../checkpoints/roberta", num_labels=3)
    if args.data == "stsb":
        model = AutoModelForSequenceClassification.from_pretrained(
            "../checkpoints/roberta", num_labels=1)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            "../checkpoints/roberta", num_labels=2)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "microsoft/deberta-base", num_labels=2) ##########

    model.config.pad_token_id = model.config.eos_token_id

    ### make model parameters non-trainable
    for param in model.parameters():
        param.requires_grad = False

    ### print the model structure to see which layers needs to be replaced by loRATT
    # print(model)

    


    tt_shape_768_768 = get_tensor_shape(config["shapes"])
    tt_rank = get_tt_rank(config["ranks"], tt_shape_768_768)


    loratt_alpha=config["alpha"]
    lorett_dropout = 0.05
    loratt_query = True
    lorett_value = True

    # layers = []

    # assign_lora = partial(LoRALinearWrapper, rank=lora_r, alpha=lora_alpha)

    #assign_lorett is a function with predefined arguments for LoRATTLinearWrapper
    #when this is called this acts as a function
    assign_lorett = partial(LoRATTLinearWrapper, tt_shape = tt_shape_768_768, tt_rank=tt_rank, alpha=loratt_alpha)

    for layer in model.roberta.encoder.layer:
        if loratt_query:
            layer.attention.self.query = assign_lorett(layer.attention.self.query, 0)
        if lorett_value:
            layer.attention.self.value = assign_lorett(layer.attention.self.value, 2)
   
    # print(model)

    # Check if linear layers are frozen
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # print("Total number of trainable parameters:", count_parameters(model))

    #for trainig and evaluation
    lightning_model = CustomLightningModule(model,args.data, config["learning_rate"])


    # callbacks = [
    #     ModelCheckpoint(
    #         save_top_k=1, mode="max", monitor="val_acc"
    #     )  # save top 1 model
    # ]
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
        )

    model_checkpoint_callback=ModelCheckpoint(
        save_top_k=1, mode="max", monitor="val_acc"
    )  

    # name="my-model" + str(args.ranks)
    # logger = CSVLogger(save_dir="logs/", name=name)


    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        accelerator="gpu",
        precision="16-mixed",
        devices=args.gpus,
        # logger=logger,
        log_every_n_steps=10,
    )



    import time
    start = time.time()

    trainer.fit(model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

    train_acc = trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best", verbose=False)
    val_acc=trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best", verbose=False)

    # print("training accuracy",train_acc)
    # print("validation accuracy",val_acc)

    print(val_acc[0]['accuracy']) #because this is what lorett reports
    # print(test_loader)
    print(type(val_acc))
    train_params=count_parameters(model)

    # log_file_name = args.data + '_' + 'log_file'

    # with open(log_file_name, 'a') as f:
    #     f.write(f'Accuracy for Tensor Dimension {args.shape} , rank {args.ranks} is: {val_acc}\n')
    #     f.write(f'Number of trainable parameters for Tensor Dimension {args.shape} , rank {args.ranks} is: {train_params}\n')
    return {"val_acc": val_acc[0]['accuracy'], "trainable_params": train_params}


def main():
    '''config = {
    # "data": tune.grid_search(["mnli", "sst2", "mrpc", "cola", "qnli", "qqp", "rte", "stsb"]),
    # "shapes": tune.choice([[12, 8, 8, 3, 8, 8, 12], [12, 8, 8, 24, 8, 12],[6, 12, 16, 16, 96], [4, 6, 6, 8, 8, 8, 24], [4, 4, 9, 12, 32, 32], [3, 4, 4, 4, 6, 32, 48]]),
    "shapes": tune.grid_search([[64, 16, 9, 64], [12, 8, 8, 8, 8, 12], [12, 8, 8, 2, 4, 8, 12], [12, 8, 8, 2, 2, 2, 8, 12], [8, 6, 2, 2, 4, 4, 2, 2, 6, 8], [8, 6, 2, 2, 2, 2, 2, 2, 2, 2, 6, 8]]),
    "ranks": tune.grid_search([1,2,4, 8, 10, 12, 16]),
    "alpha": tune.grid_search([1, 2, 4, 8, 10, 12, 16, 32]),
    # "batch_size": tune.choice([8, 16, 32]),
    "learning_rate": tune.grid_search([1e-5, 1e-4, 5e-5, 5e-4]),
}'''
    config = {
        "shapes": [12, 8, 8, 2, 2, 2, 8, 12], 
        "ranks": 2,
        "alpha": 8,
        "learning_rate":  1e-3,
    }




    analysis =  train_without_ray(config)
   
    # print(analysis)

    #save result of all tasks
    # df = analysis
    df = pd.DataFrame.from_dict(analysis, orient='index',  columns=['value']) #changed
    print(df)
    filename= f"{args.data}_ray_tune_results_roberta.csv"
    df.to_csv(filename, index=False)

   

    #save the best hyperparameters
    best_config = analysis.get('best_config', 'No best config found') #changed
    print(best_config)
    filename_best= f"{args.data}_best_hyper_roberta.txt"
    with open(filename_best,"w") as f:
        # f.write(str(analysis.best_config))
        f.write(str(best_config))

if __name__=="__main__":
    #setup_ray_cluster()
    parser = ArgumentParser()
    # parser.add_argument("--ranks", type=int, default=5)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--data", type=str, default = 'cola')
    # parser.add_argument("--shape", type=int, default=7)
    # parser.add_argument("--bs", type=int, default=128)
    args = parser.parse_args()
    #main()
    main()

