__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"

import numpy as np
import torch

from torch.utils.data import Dataset

class pretext_data(Dataset):

    def __init__(self, config, filepath):
        super(pretext_data, self).__init__()

        self.file_path = filepath
        self.idx = np.array(range(len(self.file_path)))
        self.config = config

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):

        path = self.file_path[index]
        data = np.load(path)
        pos = torch.tensor(data["pos"][:, :1, :])  # (7, 1, 3000)
        y = torch.tensor(data["y"])  # (7,)

        return pos, y  # (7, 2, 3000)


class TuneDataset(Dataset):
    """Dataset for train and test"""

    def __init__(self, subjects):
        self.subjects = subjects
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
