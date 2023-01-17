import wandb
import numpy as np
import pandas as pd
import torch
import os, argparse

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


EXPERIMENT_NAME = 'supervised_hmc'
MODALITY = 'eeg'
DATASET_PATH = '/scratch/hmc'
DATASET_FILES = os.listdir(os.path.join(DATASET_PATH, 'subjects_data'))
SAVE_PATH = './saved_weights'


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








PRETEXT_FILE = os.listdir(os.path.join(config.src_path, "pretext"))
PRETEXT_FILE.sort(key=natural_keys)
PRETEXT_FILE = [
    os.path.join(config.src_path, "pretext", f) for f in PRETEXT_FILE
]

TEST_FILE = os.listdir(os.path.join(config.le_path))
TEST_FILE.sort(key=natural_keys)
TEST_FILE = [os.path.join(config.le_path, f) for f in TEST_FILE]

print(f"Number of pretext files: {len(PRETEXT_FILE)}")
print(f"Number of test records: {len(TEST_FILE)}")

pretext_loader = DataLoader(
    pretext_data(config, PRETEXT_FILE),
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=8,
)

test_records = [np.load(f) for f in TEST_FILE]
test_subjects = dict()

for i, rec in enumerate(test_records):
    if rec["_description"][0] not in test_subjects.keys():
        test_subjects[rec["_description"][0]] = [rec]
    else:
        test_subjects[rec["_description"][0]].append(rec)

test_subjects = list(test_subjects.values())

model = sleep_pretrain(config, name, pretext_loader, test_subjects, ss_wandb)
ss_wandb.watch([model], log="all", log_freq=500)

model.fit()
ss_wandb.finish()
