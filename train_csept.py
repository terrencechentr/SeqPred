import os
import wandb
import pandas as pd
import torch
from model.configuration_csept import CseptConfig
from model.modeling_csept import CspetForSequencePrediction
from argparse import ArgumentParser
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.random import set_all_seeds
from datetime import datetime
from data.spetDataset import SpetDataset

os.environ["WANDB_START_METHOD"] = "thread"


def init_wandb(args):
    wandb_run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=args
    )
    return wandb_run

def main(args):

    print(args)

    wandb_run = init_wandb(args)

    train_dataset = SpetDataset(csv_file='/SeqPred/data/data.csv', pred_length=args.pred_length, is_train=True, noise_level=args.train_noise_level)
    test_dataset = SpetDataset(csv_file='/SeqPred/data/data.csv', pred_length=args.pred_length, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    input_size = train_dataset[0][0].shape[-1]
    
    device = 'cuda'
    model_config = CseptConfig(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        attention_dropout=args.attention_dropout,
    )
    model = CspetForSequencePrediction(model_config).to(device)
    
    optimizer = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adagrad": torch.optim.Adagrad,
    }[args.optimizer](model.parameters(), lr=args.learning_rate)


    for epoch in range(args.epochs):
        
        # train
        model.train()
        train_loss = 0
        for input_values, labels in train_loader:
            input_values, labels = input_values.to(device), labels.to(device)
            output = model(input_values=input_values, labels=labels)
            
            loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
                
        wandb_run.log({'train_loss': train_loss / len(train_loader)})
        print(f"Epoch {epoch+1} train loss: {train_loss / len(train_loader)}")
        

        # test
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for input_values, labels in test_loader:
                input_values, labels = input_values.to(device), labels.to(device)
                output = model(input_values=input_values, labels=labels)
                loss = output.loss
                test_loss += loss.item()
        wandb_run.log({'test_loss': test_loss / len(test_loader)})
        print(f"Epoch {epoch+1} test loss: {test_loss / len(test_loader)}")


if __name__ == "__main__":
    set_all_seeds(42)

    argparser = ArgumentParser()
    argparser.add_argument("--wandb_entity", type=str, default="terrencechen")
    argparser.add_argument("--wandb_project", type=str, default="SeqPred-grid-test")
    argparser.add_argument("--wandb_run_name", type=str, default=f"csept_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    argparser.add_argument("--learning_rate", type=float, default=1e-3)
    argparser.add_argument("--epochs", type=int, default=50)
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--hidden_size", type=int, default=8)
    argparser.add_argument("--num_hidden_layers", type=int, default=8)
    argparser.add_argument("--num_attention_heads", type=int, default=2)
    argparser.add_argument("--num_key_value_heads", type=int, default=2)
    argparser.add_argument("--train_noise_level", type=float, default=0.01)
    argparser.add_argument("--pred_length", type=int, default=256)
    argparser.add_argument("--attention_dropout", type=float, default=0.1)
    argparser.add_argument("--optimizer", type=str, default="adam")
    args = argparser.parse_args()

    main(args)