__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"

import numpy as np
from torch.utils.data import Dataset

class distillDataset(Dataset):
    """Dataset for train and test"""

    def __init__(self, filepath):
        super(distillDataset, self).__init__()
        self.file_path = filepath

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        path = self.file_path[index]
        data = np.load(path)
        return data[self.modality], data['y']