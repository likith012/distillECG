import os
import numpy as np
import argparse
from tqdm import tqdm
import multiprocessing

SEED = 1234
rng = np.random.RandomState(SEED)


# ARGS
HALF_WINDOW = 3 # Epoch length is HALF_WINDOW*2 + 1
NUM_CORES = multiprocessing.cpu_count()
AVAILABLE_MODALITY = ['eeg', 'ecg', 'eog', 'emg']

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="/scratch/shhs",
                    help="File path to the PSG and annotation files.")
args = parser.parse_args()

DATASET_SUBJECTS = sorted(os.listdir(os.path.join(args.dir, 'subjects_data')))
DATASET_SUBJECTS = [os.path.join(args.dir, 'subjects_data', f) for f in DATASET_SUBJECTS]
TRAIN_PATH = os.path.join(args.dir, f'train_{HALF_WINDOW*2 +1}') 
TEST_PATH = os.path.join(args.dir, f'test_{HALF_WINDOW*2 + 1}')

train_subjects = rng.choice(DATASET_SUBJECTS, int(len(DATASET_SUBJECTS)*0.8), replace=False)
test_subjects = list(set(DATASET_SUBJECTS) - set(train_subjects))

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"Train subjects: {len(train_subjects)} \n")
print(f"Test subjects: {len(test_subjects)} \n")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

if not os.path.exists(TRAIN_PATH): os.makedirs(TRAIN_PATH, exist_ok=True)
if not os.path.exists(TEST_PATH): os.makedirs(TEST_PATH, exist_ok=True)

def preprocess_subjects(subject_paths, save_path, k, N):
    subjects_data = [np.load(f) for i, f in enumerate(subject_paths) if i%N==k]
    cnt = 0
    for file in tqdm(subjects_data, desc="Data processing ...", total=len(subjects_data)):
        y = file["y"].astype('int')
        eeg = file['eeg']
        eog = file['eog']
        emg = file['emg']
        ecg = file['ecg']
        num_epochs = file["epoch_length"]

        for i in range(HALF_WINDOW, num_epochs-HALF_WINDOW):
            epochs_data = {}
            temp_path = os.path.join(save_path, f"{k+1}_{cnt}.npz")
            epochs_data['eeg'] = eeg[i-HALF_WINDOW: i+HALF_WINDOW+1]
            epochs_data['eog'] = eog[i-HALF_WINDOW: i+HALF_WINDOW+1]
            epochs_data['ecg'] = ecg[i-HALF_WINDOW: i+HALF_WINDOW+1]
            epochs_data['emg'] = emg[i-HALF_WINDOW: i+HALF_WINDOW+1]
            epochs_data['y'] = y[i-HALF_WINDOW: i+HALF_WINDOW+1]
            np.savez(temp_path, **epochs_data)
            cnt+=1

p_list = []
for k in range(NUM_CORES):
    process = multiprocessing.Process(target=preprocess_subjects, args=(train_subjects, TRAIN_PATH, k, NUM_CORES))
    process.start()
    p_list.append(process)
for i in p_list:
    i.join()

p_list = []
for k in range(NUM_CORES):
    process = multiprocessing.Process(target=preprocess_subjects, args=(test_subjects, TEST_PATH, k, NUM_CORES))
    process.start()
    p_list.append(process)
for i in p_list:
    i.join()