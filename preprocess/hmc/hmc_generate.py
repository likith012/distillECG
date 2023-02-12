import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing

import mne
from braindecode.preprocessing.preprocess import preprocess, Preprocessor, zscore
from braindecode.preprocessing.windowers import create_windows_from_events
from braindecode.datasets import BaseDataset, BaseConcatDataset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
mne.set_log_level(verbose='WARNING')


HMC_PATH = '/scratch/hmc/recordings'
HMC_SAVE_PATH = os.path.join(os.path.split(HMC_PATH)[0], 'subjects_data')
NUM_CORES = multiprocessing.cpu_count()

if not os.path.exists(HMC_SAVE_PATH):
    os.makedirs(HMC_SAVE_PATH, exist_ok=True)

window_size = 30
sfreq = 100
window_size_samples = window_size*sfreq
subject_ids = range(1, 155)

hmc_files = os.listdir(HMC_PATH)
raw_paths, ann_paths = [], []
for subject in subject_ids:
    current_file = f"SN{subject:03d}.edf"
    if  current_file in hmc_files:
        raw_paths.append(os.path.join(HMC_PATH, current_file))
        ann_paths.append(os.path.join(HMC_PATH, f"SN{subject:03d}_sleepscoring.edf"))


label_mapping = {  
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 1,
    "Sleep stage N3": 2,
    "Sleep stage R": 3,
}
channel_mapping = {
    'eeg': ['EEG'],
    'ecg': ['ECG'],
    'eog': ['EOG(L)', 'EOG(R)'],
    'emg': ['EMG'],
}

class HMCSleepStaging(BaseConcatDataset):
    
    def __init__(
        self,
        raw_path=None,
        ann_path=None,
        channels=None,
        preload=False,
        crop_wake_mins=0,
        crop=None,
    ):
        if (raw_path is None) or (ann_path is None) or (channels is None):
            raise Exception("Please provide paths for raw and annotations file!")

        raw, desc = self._load_raw(
            raw_path,
            ann_path,
            channels,
            preload=preload,
            crop_wake_mins=crop_wake_mins,
            crop=crop
        )
        base_ds = BaseDataset(raw, desc)
        super().__init__([base_ds])

    def _load_raw(
        self,
        raw_fname,
        ann_fname,
        channels,
        preload,
        crop_wake_mins,
        crop,
    ):
        raw = mne.io.read_raw_edf(raw_fname, preload=preload, include=channels)
        annots = mne.read_annotations(ann_fname)
        raw.set_annotations(annots, emit_warning=False)
        raw.resample(sfreq, npad="auto")

        if crop_wake_mins > 0:
            # Find first and last sleep stages
            mask = [x[-1] in ["1", "2", "3", "R"] for x in annots.description]
            sleep_event_inds = np.where(mask)[0]

            # Crop raw
            tmin = annots[int(sleep_event_inds[0])]["onset"] - crop_wake_mins * 60
            tmax = annots[int(sleep_event_inds[-1])]["onset"] + crop_wake_mins * 60
            raw.crop(tmin=max(tmin, raw.times[0]), tmax=min(tmax, raw.times[-1]))

        if crop is not None:
            raw.crop(*crop)

        raw_basename = os.path.basename(raw_fname)
        subj_nb = int(raw_basename[2:5])
        desc = pd.Series({"subject_id": subj_nb,}, name="")
        return raw, desc

    
def __get_epochs(windows_subject):
    epochs_data = []
    for epoch in windows_subject.windows:
        epochs_data.append(epoch)
    epochs_data = np.stack(epochs_data, axis=0) # Shape of (num_epochs, num_channels, num_sample_points)
    return epochs_data

def __get_channels(raw, ann):
    channels_data = dict()
    for ch in channel_mapping.keys():
        subject_dataset = HMCSleepStaging(raw_path=raw, ann_path=ann, channels=channel_mapping[ch], preload=True)

        windows_subject_dataset = create_windows_from_events(
                                subject_dataset,
                                window_size_samples=window_size_samples,
                                window_stride_samples=window_size_samples,
                                preload=True,
                                mapping=label_mapping,
                            )
        preprocess(windows_subject_dataset, [Preprocessor(zscore)])

        channels_data[ch] = __get_epochs(*windows_subject_dataset.datasets)
    channels_data['y'] = windows_subject_dataset.datasets[0].y
    channels_data['subject_id'] = windows_subject_dataset.datasets[0].description['subject_id']
    channels_data['epoch_length'] = len(windows_subject_dataset.datasets[0])
    return channels_data, windows_subject_dataset.datasets[0].description['subject_id']


def preprocess_dataset(raw_paths, ann_paths, k, N):
    raw_paths_core = [f for i, f in enumerate(raw_paths) if i%N==k]
    ann_paths_core = [f for i, f in enumerate(ann_paths) if i%N==k]
    for raw, ann in tqdm(zip(raw_paths_core, ann_paths_core), desc="HMC dataset preprocessing ...", total=len(raw_paths_core)):
        channels_data, subject_num = __get_channels(raw, ann)    
        subjects_save_path = os.path.join(HMC_SAVE_PATH, f"{subject_num}.npz")
        np.savez(subjects_save_path, **channels_data)

p_list = []
for k in range(NUM_CORES):
    process = multiprocessing.Process(target=preprocess_dataset, args=(raw_paths, ann_paths, k, NUM_CORES))
    process.start()
    p_list.append(process)
for i in p_list:
    i.join()
    