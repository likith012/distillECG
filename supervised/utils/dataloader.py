__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"

import numpy as np
import torch
from torch.utils.data import Dataset


class distillDataset(Dataset):
    """Dataset for train and test"""

    def __init__(self, subjects, metadata_df, modality, epoch_len):
        self.subjects = subjects
        self.metadata = metadata_df
        self.epoch_len = epoch_len
        self.modality = modality

        self._add_subjects()

    def __getitem__(self, index):

        X = self.X[index, :1, :]
        y = self.y[index]
        return X, y

    def __len__(self):
        return self.X.shape[0]

    def _add_subjects(self):
        self.X = []
        self.y = []
        for subject in self.subjects:
            self.X.append(subject["windows"])
            self.y.append(subject["y"])
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)



class distill(Dataset):
    """Dataset for train and test"""

    def __init__(self, filepath):
        super(distill, self).__init__()

        self.file_path = filepath

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):

        path = self.file_path[index]
        data = np.load(path)
        return data['X'], data['y']