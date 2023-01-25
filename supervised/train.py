import wandb, os
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.utils import *
from utils.dataloader import distillDataset
from helper_train import distill_train

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

DATASET = 'shhs'
MODALITY = 'eeg'
EXPERIMENT_NAME = f'supervised_{DATASET}+{MODALITY}'
INPUT_CHANNELS = 1
NUM_CLASSES = 4
BATCH_SIZE = 256
EPOCH_LEN = 7

DATASET_PATH = f'/scratch/{DATASET}'
DATASET_SUBJECTS = os.listdir(os.path.join(DATASET_PATH, 'subjects_data'))
SAVE_PATH = './saved_weights'
if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH, exist_ok=True)

wandb = wandb.init(
    project="distillECG",
    name=EXPERIMENT_NAME,
    save_code=False,
    entity="sleep-staging",
)
wandb.save("./supervised/utils/*")
wandb.save("./supervised/models/*")
wandb.save("./supervised/helper_train.py")
wandb.save("./supervised/train.py")

DATASET_SUBJECTS.sort(key=natural_keys)
DATASET_SUBJECTS = [os.path.join(DATASET_PATH, 'subjects_data', f) for f in DATASET_SUBJECTS]
dataset_subjects_data = [np.load(f) for f in DATASET_SUBJECTS]

# load files
TRAIN_PATH = os.path.join(DATASET_PATH, f'train_{EPOCH_LEN}') 
TEST_PATH = os.path.join(DATASET_PATH, f'test_{EPOCH_LEN}')
TRAIN_EPOCH_FILES = [os.path.join(TRAIN_PATH, f) for f in os.listdir(TRAIN_PATH)]
TEST_EPOCH_FILES = [os.path.join(TEST_PATH, f) for f in os.listdir(TEST_PATH)]


print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"Train files: {len(TRAIN_EPOCH_FILES)} \n")
print(f"Test files: {len(TEST_EPOCH_FILES)} \n")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

train_dataset = distillDataset(TRAIN_EPOCH_FILES, MODALITY)
test_dataset = distillDataset(TEST_EPOCH_FILES, MODALITY)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
)

model = distill_train(EXPERIMENT_NAME, SAVE_PATH, train_loader, test_loader, wandb, EPOCH_LEN, INPUT_CHANNELS, NUM_CLASSES)
wandb.watch([model], log="all", log_freq=500)

model.fit()
wandb.finish()
