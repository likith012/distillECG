import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import mne

from braindecode.preprocessing.preprocess import preprocess, Preprocessor, zscore
from braindecode.preprocessing.windowers import create_windows_from_events
from braindecode.datasets import BaseConcatDataset, BaseDataset
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


HMC_PATH = '/scratch/hmc/recordings'

window_size = 30
sfreq = 100
window_size_samples = window_size*sfreq
mapping = {  
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
}


class HMCSleepStaging(BaseConcatDataset):
    
    def __init__(
        self,
        hmc_path=None,
        subject_ids=None,
        preload=False,
        crop_wake_mins=0,
        crop=None,
    ):
        if subject_ids is None:
            subject_ids = range(1, 155)
        if hmc_path is None:
            raise Exception("Please provide path")
        
        self.raw_files, self.edf_files = [], []       
        self._fetch_data(subject_ids, hmc_path)

        all_base_ds = list()
        for raw_fname, ann_fname in zip(self.raw_files, self.edf_files):
            raw, desc = self._load_raw(
                raw_fname,
                ann_fname,
                preload=preload,
                crop_wake_mins=crop_wake_mins,
                crop=crop
            )
            base_ds = BaseDataset(raw, desc)
            all_base_ds.append(base_ds)
        super().__init__(all_base_ds)
    
    def _fetch_data(
        self,
        subject_ids,
        hmc_path,
    ):
        hmc_files = os.listdir(hmc_path)
        for subject in subject_ids:
            current_file = f"SN{subject:03d}.edf"
            if  current_file in hmc_files:
                self.raw_files.append(os.path.join(hmc_path, current_file))
                self.edf_files.append(os.path.join(hmc_path, f"SN{subject:03d}_sleepscoring.edf"))        
    
    @staticmethod
    def _load_raw(
        raw_fname,
        ann_fname,
        preload,
        crop_wake_mins,
        crop,
    ):
        raw = mne.io.read_raw_edf(raw_fname, preload=preload)
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
        epoch_data.append(epoch)
    epochs_data = np.stack(epochs_data, axis=0) # Shape of (num_epochs, num_channels, num_sample_points)
    return epochs_data


hmc_dataset = HMCSleepStaging(hmc_path=HMC_PATH)
hmc_windows_dataset = create_windows_from_events(
                        hmc_dataset,
                        window_size_samples=window_size_samples,
                        window_stride_samples=window_size_samples,
                        preload= True,
                        mapping=mapping,
                        n_jobs=-1
                    )
preprocess(windows_dataset, [Preprocessor(zscore)])


metadata_df = pd.DataFrame(columns=['subject_id', 'epoch_length'])
for windows_subject in tqdm(hmc_windows_dataset.datasets, desc="HMC dataset preprocessing ..."):
    metadata_df = metadata_df.append({'subject_id': windows_subject.description['subject_id'], 'epoch_length': len(windows_subject)}, 
                                     ignore_index=True)

    hmc_subject_data = __get_epochs(windows_subject)
    subjects_save_path = os.path.join(os.path.split(HMC_PATH)[0], 'hmc_subjects', f"{windows_subject.description['subject_id']:03d}.npz")
    np.savez(subjects_save_path, 
             eeg=hmc_subject_data[:4], 
             emg=hmc_subject_data[4:5],
             eog=hmc_subject_data[5:7],
             emog=hmc_subject_data[4:7],
             ecg=hmc_subject_data[7:],
             y=windows_subject.y
            )
metadata_df.to_csv(os.path.join(os.path.split(HMC_PATH)[0], 'metadata.csv'))
    