import os
import random
import torch
import numpy as np
import torch.nn.functional as F
import cv2

splits_all_path = '/media/mountHDD1/LanxHuyen/CVPR2017/block_splits_by_image_all.pth' 
eeg_path = '/media/mountHDD1/LanxHuyen/CVPR2017/eeg_55_95_std.pth'
output_path_dir = '/media/mountHDD1/LanxHuyen/CVPR2017'

splits_all = torch.load(splits_all_path)
eeg_loaded = torch.load(eeg_path)
dataset, labels, images = [eeg_loaded[k] for k in eeg_loaded.keys()]

# Extract EEG data from the dictionary list
# eeg_tensors = [sample['eeg'] for sample in dataset]

# Method 1: Creating grayscale heatmaps for each trial
def process_method_1(eeg_data, time_low=20, time_high=460, output_shape=(512, 440)):
    """
    
    """
    eeg_data = eeg_data[:, time_low:time_high]
    normalized_data = (eeg_data - eeg_data.min()) / (eeg_data.max() - eeg_data.min())
    grayscale_images = (normalized_data * 255).to(torch.uint8)
    grayscale_images = grayscale_images.unsqueeze(0).unsqueeze(0) # (1, 1, h, w)
    resized_images = F.interpolate(grayscale_images, size=output_shape, mode='bilinear', align_corners=True)
    resized_images = resized_images.squeeze(0).squeeze(0)
    resized_images = torch.tensor(resized_images, dtype=torch.float32)
    return resized_images
# Apply method 1 to each tensor in eeg_tensors
for sample in dataset:
    sample['eeg'] = process_method_1(sample['eeg'])


