import os
import numpy as np
import argparse
from tqdm import tqdm


SEED = 1234
np.random.seed(SEED)
rng = np.random.RandomState(SEED)


# ARGS
HALF_WINDOW = 3 # Epoch length is HALF_WINDOW*2 + 1
AVAILABLE_MODALITY = ['eeg', 'ecg', 'eog', 'emg', 'emog']

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="/scratch/shhs",
                    help="File path to the PSG and annotation files.")
args = parser.parse_args()

DATASET_SUBJECTS = sorted(os.listdir(os.path.join(args.dir, 'subjects_data')))
DATASET_SUBJECTS = [os.path.join(args.dir, 'subjects_data', f) for f in DATASET_SUBJECTS]
TRAIN_PATH = os.path.join(args.dir, f'train_{HALF_WINDOW}') 
TEST_PATH = os.path.join(args.dir, f'test_{HALF_WINDOW}')

dataset_subjects_data = [np.load(f) for f in DATASET_SUBJECTS]
sub_train = rng.choice(DATASET_SUBJECTS, int(len(DATASET_SUBJECTS)*0.8), replace=False)
sub_test = list(set(DATASET_SUBJECTS) - set(sub_train))
train_data = [np.load(f) for f in sub_train]
test_data = [np.load(f) for f in sub_test]

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"Train subjects: {len(sub_train)} \n")
print(f"Test subjects: {len(sub_test)} \n")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

if not os.path.exists(TRAIN_PATH): os.makedirs(TRAIN_PATH, exist_ok=True)
if not os.path.exists(TEST_PATH): os.makedirs(TEST_PATH, exist_ok=True)

cnt = 0
for file in tqdm(train_data, desc="Train data processing ..."):
    y = file["y"].astype('int')
    num_epochs = file["epoch_length"]

    for i in range(HALF_WINDOW, num_epochs-HALF_WINDOW):
        epochs_data = {}
        temp_path = os.path.join(TRAIN_PATH, str(cnt)+".npz")
        for modality in AVAILABLE_MODALITY:
            epochs_data[modality] = file[modality][i-HALF_WINDOW: i+HALF_WINDOW+1]
        epochs_data['y'] = y[i-HALF_WINDOW: i+HALF_WINDOW+1]
        np.savez(temp_path, **epochs_data)
        cnt+=1

cnt = 0
for file in tqdm(test_data, desc="Test data processing ..."):
    y_dat = file["y"].astype('int')
    num_epochs = file["epoch_length"]

    for i in range(HALF_WINDOW, num_epochs-HALF_WINDOW):
        epochs_data = {}
        temp_path = os.path.join(TEST_PATH, str(cnt)+".npz")
        for modality in AVAILABLE_MODALITY:
            epochs_data[modality] = file[modality][i-HALF_WINDOW: i+HALF_WINDOW+1]
        epochs_data['y'] = y[i-HALF_WINDOW: i+HALF_WINDOW+1]
        np.savez(temp_path, **epochs_data)
        cnt+=1