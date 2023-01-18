import wandb, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.utils import *
from utils.dataloader import distillDataset, distill
from helper_train import distill_train

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
rng = np.random.RandomState(SEED)

from tqdm import tqdm

EXPERIMENT_NAME = 'supervised_hmc'
MODALITY = 'eeg'
DO_KFOLD = False
BATCH_SIZE = 256
EPOCH_LEN = 7

DATASET_PATH = '/scratch/hmc'
DATASET_SUBJECTS = os.listdir(os.path.join(DATASET_PATH, 'subjects_data'))
SAVE_PATH = './saved_weights'
metadata_df = pd.load_csv(os.listdir(os.path.join(DATASET_PATH, 'metadata.csv')))


wandb = wandb.init(
    project="distillECG",
    name=EXPERIMENT_NAME,
    save_code=False,
    entity="sleep-staging",
)

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)

wandb.save("./supervised/utils/*")
wandb.save("./supervised/models/*")
wandb.save("./supervised/helper_train.py")
wandb.save("./supervised/train.py")

DATASET_SUBJECTS.sort(key=natural_keys)
DATASET_SUBJECTS = [os.path.join(DATASET_PATH, 'subjects_data', f) for f in DATASET_SUBJECTS]
dataset_subjects_data = [np.load(f) for f in DATASET_SUBJECTS]


sub_train = rng.choice(DATASET_SUBJECTS, int(len(DATASET_SUBJECTS)*0.8), replace=False)
sub_test = sorted(list(set(DATASET_SUBJECTS) - set(sub_train)))

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"Train: {len(sub_train)} \n")
print(f"Test: {len(sub_test)} \n")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

TRAIN_FILES = [np.load(f) for f in sub_train]
TEST_FILES = [np.load(f) for f in sub_test]

# load files
TRAIN_PATH = os.path.join(DATASET_PATH, 'train_eeg') 
TEST_PATH = os.path.join(DATASET_PATH, 'test_eeg')

if not os.path.exists(TRAIN_PATH):
    os.makedirs(TRAIN_PATH, exist_ok=True)

if not os.path.exists(TEST_PATH):
    os.makedirs(TEST_PATH, exist_ok=True)

half_window = 3

cnt = 0
for file in tqdm(TRAIN_FILES):
    x_dat = file[MODALITY]
    y_dat = file["y"].astype('int')

    for i in range(half_window,x_dat.shape[0]-half_window):
        dct = {}
        temp_path = os.path.join(TRAIN_PATH, str(cnt)+".npz")
        dct['X'] = x_dat[i-half_window:i+half_window+1]
        dct['y'] = y_dat[i-half_window:i+half_window+1]
        np.savez(temp_path,**dct)
        cnt+=1

cnt = 0
for file in tqdm(TEST_FILES):
    x_dat = file[MODALITY]
    y_dat = file["y"].astype('int')

    for i in range(half_window,x_dat.shape[0]-half_window):
        dct = {}
        temp_path = os.path.join(TEST_PATH, str(cnt)+".npz")
        dct['X'] = x_dat[i-half_window:i+half_window+1]
        dct['y'] = y_dat[i-half_window:i+half_window+1]
        np.savez(temp_path,**dct)
        cnt+=1

TRAIN_EPOCH_FILES = [os.path.join(TRAIN_PATH, f) for f in os.listdir(TRAIN_PATH)]
TEST_EPOCH_FILES = [os.path.join(TEST_PATH, f) for f in os.listdir(TEST_PATH)]

train_dataset = distill(TRAIN_EPOCH_FILES)
test_dataset = distill(TEST_EPOCH_FILES)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
)

model = distill_train(EXPERIMENT_NAME, train_loader, test_loader, wandb, EPOCH_LEN, MODALITY)
wandb.watch([model], log="all", log_freq=500)

model.fit()
wandb.finish()
