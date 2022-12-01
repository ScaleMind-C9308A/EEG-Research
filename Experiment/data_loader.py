import torch 
import random
import copy

class EEGDataset:
    def __init__(self, iv, input_path, classifier, map_idx=None):
        self.iv = iv
        # Load EEG signals
        loaded = torch.load(input_path)
        self.data = loaded["dataset"]
        self.classifier = classifier
        self.size = len(self.data) # 40000
    def __getitem__(self, i):
        # Get EEG 
        eeg = self.data[i]["eeg"]
        # Get label
        label = self.data[i]["label"]
        return eeg, label
    def __len__(self):
        return self.size

class Splitter:
    def __init__(self,
                iv,
                dataset,
                splits_path,
                classes,
                split_num,
                split_name):
        # Load split
        loaded = torch.load(splits_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # filter only data in chosen classes
        self.split_idx = [i for i in self.split_idx 
                            if dataset.data[i]["label"] in classes]
        # Compute size
        self.size = len(self.split_idx)
    def __len__(self):
        return self.size
    def __getitem__(self, i):
        return self.split_idx[i]

class SplitterWithData:
    def __init__(self,
                iv,
                dataset,
                splits_path,
                classes,
                split_num,
                split_name,
                relabel):
        # Set EEG dataset
        self.dataset = dataset
        self.classes = classes
        self.relabel = relabel
        # Load split
        loaded = torch.load(splits_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # filter only data in chosen classes
        self.split_idx = [i for i in self.split_idx 
                            if dataset.data[i]["label"] in classes]
        # Compute size
        self.size = len(self.split_idx)
    def __len__(self):
        return self.size
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        if self.relabel:
            label = self.classes.index(label)
        return eeg, label