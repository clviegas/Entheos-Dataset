"""
Written by Carla Viegas


"""

import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader


class Concat_CSVDataset(Dataset):
    """Dataset class that can load data from one main_csv file that contains filename, features and labels."""
    def __init__(self, X, feature_path_list):
        """
        Args:
            csv_file (string): Path to the csv file with filename per data split.
            feature_csv (string) : Path to the csv file with filename, feature and labels.

        """
        self.filenames = X
        self.features = []

        for arg in feature_path_list:
            self.features.append(pd.read_csv(arg, index_col=0))

    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.filenames[idx]

        feature = []
        for df in self.features:
            feature.extend(np.array(df.loc[filename][0:-1]))
        feature = np.array(feature)
        label = self.features[0].loc[filename].loc['label']
        label = label.astype('float')
        sample = (feature,label)

        return sample
