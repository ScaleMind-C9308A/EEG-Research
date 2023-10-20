from scipy import signal
import os
import random
import torch
import numpy as np
import torch.nn.functional as F

splits_all_path = '/media/mountHDD1/LanxHuyen/CVPR2017/block_splits_by_image_all.pth' 
eeg_path = '/media/mountHDD1/LanxHuyen/CVPR2017/eeg_55_95_std.pth'
output_path_dir = '/media/mountHDD1/LanxHuyen/CVPR2017'
output_filename = 'eeg_55_95_std_downsampled_128Hz.pth'

splits_all = torch.load(splits_all_path)
eeg_loaded = torch.load(eeg_path)
eeg_dataset, labels, images = [eeg_loaded[k] for k in eeg_loaded.keys()]

def eeg_downsample(eeg_data, time_low=20, time_high=460):
    """
    Downsample EEG data from 440Hz to 128Hz.
    """
    eeg_data = eeg_data[:, time_low:time_high] # (128, 440) => cut eeg signal to 440 time steps
    eeg_data = np.array(eeg_data) # (128, 440)
    # Resample from 440Hz to 128Hz
    eeg_data = signal.resample_poly(eeg_data, up=128, down=440, axis=1, padtype='line') # (128, 128)
    # Convert to torch tensor
    eeg_data = torch.tensor(eeg_data, dtype=torch.float32) # (128, 128)
    return eeg_data

print(f"Before downsample: Size of an EEG sample (tensor): {eeg_dataset[0]['eeg'].size()}")
for sample in eeg_dataset:
    sample['eeg'] = eeg_downsample(sample['eeg'])
eeg_loaded['dataset'] = eeg_dataset
print(f"After downsample: Size of an EEG sample (tensor): {eeg_dataset[0]['eeg'].size()}")
torch.save(eeg_loaded, os.path.join(output_path_dir, output_filename))

