import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import mne
import xml.etree.ElementTree as ET
from braindecode.preprocessing.preprocess import preprocess, Preprocessor, zscore
from braindecode.preprocessing.windowers import create_windows_from_events
from braindecode.datasets import BaseDataset, BaseConcatDataset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
mne.set_log_level(verbose='WARNING')


SHHS_PATH = '/scratch/shhs/edfs/shhs1'
SHHS_EVENTS_PATH = '/scratch/shhs/annotations-events-profusion\shhs1'
SELECTED_SUBJECTS_PATH = './preprocess/shhs/selected_shhs1.txt'
SHHS_SAVE_PATH = os.path.join(os.path.split(os.path.split(SHHS_PATH)[0])[0], 'subjects_data')

if not os.path.exists(SHHS_SAVE_PATH):
    os.makedirs(SHHS_SAVE_PATH, exist_ok=True)


window_size = 30
sfreq = 100
window_size_samples = window_size*sfreq
subject_ids = pd.read_csv(SELECTED_SUBJECTS_PATH, header=None)
subject_ids = subject_ids[0].values.tolist()
raw_paths = [os.path.join(SHHS_PATH, f'{f}.edf') for f in subject_ids]
ann_paths = [os.path.join(SHHS_EVENTS_PATH, f'{f}-profusion.xml') for f in subject_ids]

label_mapping = {  
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
}
channel_mapping = {
    'eeg': ['EEG', 'EEG(sec)'],
    'ecg': ['ECG'],
    'eog': ['EOG(L)', 'EOG(R)'],
    'emg': ['EMG'],
    'emog': ['EOG(L)', 'EOG(R)', 'EMG'],
}

class SHHSSleepStaging(BaseConcatDataset):
    
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

    @staticmethod    
    def read_annotations(ann_fname):
        labels = []
        t = ET.parse(ann_fname)
        r = t.getroot()

        for i in range(len(r[4])):
            lbl = int(r[4][i].text)
            if lbl == 0:
                labels.append("Sleep stage W")
            elif lbl == 1:
                labels.append("Sleep stage N1")
            elif lbl == 2:
                labels.append("Sleep stage N2")
            elif (lbl == 3) or (lbl == 4):
                labels.append("Sleep stage N3")
            elif lbl == 5:
                labels.append("Sleep stage R")
            else:
                print( "============================== Faulty file =============================")

        labels = np.asarray(labels)
        onsets = [window_size*i for i in range(len(labels))]
        onsets = np.asarray(onsets)
        durations = np.repeat(window_size, len(labels))
        annots = mne.Annotations(onsets, durations, labels)
        return annots

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
        annots = self.read_annotations(ann_fname)
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
        subj_nb = int(raw_basename[-10:-4])
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
        subject_dataset = SHHSSleepStaging(raw_path=raw, ann_path=ann, channels=channel_mapping[ch], preload=True)
        preprocess(subject_dataset, [Preprocessor(lambda x: x * 1e6)])

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


if __name__ == "__main__":
    for raw, ann in tqdm(zip(raw_paths, ann_paths), desc="SHHS dataset preprocessing ...", total=len(raw_paths)):
        channels_data, subject_num = __get_channels(raw, ann)    
        subjects_save_path = os.path.join(SHHS_SAVE_PATH, f"{subject_num}.npz")
        np.savez(subjects_save_path, **channels_data)
    