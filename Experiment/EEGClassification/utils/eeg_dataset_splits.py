# 1947 images (erased label 33)
# We want to splits by image

import os
import random
import torch
import numpy as np

splits_all_path = '/media/mountHDD1/LanxHuyen/CVPR2017/block_splits_by_image_all.pth' 
eeg_path = '/media/mountHDD1/LanxHuyen/CVPR2017/eeg_55_95_std.pth'
output_path_dir = '/media/mountHDD1/LanxHuyen/CVPR2017'

splits_all = torch.load(splits_all_path)
eeg_loaded = torch.load(eeg_path)
dataset, labels, images = [eeg_loaded[k] for k in eeg_loaded.keys()]

def remove_label_from_splits(dataset, splits_all, label):
    removed_labels_idx = [i for i in range(len(dataset)) if dataset[i]['label'] == label] 

    train_splits, val_splits, test_splits = [splits_all['splits'][0][k] for k in splits_all['splits'][0].keys()]
    train_splits = [sample for sample in train_splits if sample not in removed_labels_idx]
    val_splits = [sample for sample in val_splits if sample not in removed_labels_idx]
    test_splits = [sample for sample in test_splits if sample not in removed_labels_idx]

    splits = {"train": train_splits, "val": val_splits, "test": test_splits}
    torch.save(splits, os.path.join(output_path_dir, "splits_by_image_removed_33.pth"))

def group_splits_by_subject(dataset, splits):
    splits = splits['splits'][0]
    result_splits = {}
    for key, split in splits.items():
        images_idx = np.array([dataset[idx]['image'] for idx in split]) # => len(images_idx) = len(split)
        grouped_images_idx = np.array([np.where(images_idx == image)[0] for image in np.unique(images_idx)]) #(split_len, 6)
        result_splits[key] = grouped_images_idx
    torch.save(result_splits, os.path.join(output_path_dir, "splits_by_subject.pth"))




