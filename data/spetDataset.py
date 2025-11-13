import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import numpy as np
class SpetDataset(Dataset):
    def __init__(
        self, 
        csv_file, 
        pred_length, 
        is_train=True, 
        noise_level=0.1,
        window_size=50
    ):
        self.is_train = is_train
        self.noise_level = noise_level
        self.pred_length = pred_length
        self.window_size = window_size
        data = pd.read_csv(csv_file)
        train_data, test_data = self.process_data(data)

        if self.is_train:
            self.data = train_data
        else:
            self.data = test_data


    def __len__(self):
        return len(self.data) - self.pred_length - 1

    def __getitem__(self, idx):
        data = self.data[idx:idx+self.pred_length+1]

        if self.is_train:
            data = data + torch.randn_like(data) * self.noise_level

        return data, data[:, 0:1]

    def process_data(self, data):

        def rolling_mean(data, window_size):
            roll_data = []
            for i in range(len(data)):
                if i < window_size:
                    roll_data.append(data[i])
                else:
                    roll_data.append(np.mean(data[i-window_size+1:i+1]))
            return roll_data

        all_data = torch.tensor(
                [
                (rolling_mean(data['Close'].pct_change().tolist()[1:], self.window_size)),
                # (data['Open'].pct_change().tolist()[1:]),
                # (data['High'].pct_change().tolist()[1:]),
                # (data['Low'].pct_change().tolist()[1:]),
                # (data['Volume'].pct_change().tolist()[1:]),
            ]
            ).transpose(0, 1).float()



        train_data = all_data[3000:4000]
        test_data = all_data[5000:6000]

        self.mean, self.std = train_data.mean(dim=0), train_data.std(dim=0)
        train_data = (train_data - self.mean) / self.std
        test_data = (test_data - self.mean) / self.std

        return train_data, test_data

