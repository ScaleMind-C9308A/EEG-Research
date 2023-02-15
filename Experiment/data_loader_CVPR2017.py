import torch 
import random
import copy

class EEGDataset:
    # Constructor
    def __init__(self, opt, eeg_signals_path):
        """
        opt: {
            subject:,
            time_low:,
            time_high:,
            model_type:
        }
        """
        self.opt = opt
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        if opt.subject!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
        else:
            self.data=loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t() #convert from (128, 500) to (500, 128)
        eeg = eeg[self.opt.time_low:self.opt.time_high,:]

        if self.opt.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(128,self.opt.time_high-self.opt.time_low)
        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label

class Splitter:
    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600] #Filter to take segments from 450 to 600 time 00steps
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label

