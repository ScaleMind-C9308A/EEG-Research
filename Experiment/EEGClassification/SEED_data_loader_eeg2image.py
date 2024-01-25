import torch
from torcheeg.datasets import SEEDIVDataset
from torcheeg import transforms as eeg_transforms

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from audiomentations import Compose, AddGaussianSNR, AddGaussianNoise, TimeStretch, PitchShift, Shift, AddGaussianSNR, Gain, GainTransition
import numpy as np
import cv2
import random

class EEG2Image_Augment_Dataset(Dataset):
    """
    EEG2Image_Augment_Dataset
    
    """
    def __init__(self, eeg_dataset, loaded_splits,time_low, time_high, mode="train", transform=None):
        """
        Args:
            img_dir_path: directory path of imagenet images,
            loaded_eeg: eeg dataset loaded from torch.load(),
            loaded_splits: cross-validation splits loaded from torch.load(),
        All arrays and data are returned as torch Tensors
        """
        self.mode = mode
        self.transform = transform

        self.time_low = time_low
        self.time_high = time_high
        # dataset, classes, img_filenames = [loaded_eeg[k] for k in ['dataset', 'labels', 'images']]
        # self.classes = classes
        # self.img_filenames = img_filenames

        self.eeg_dataset = eeg_dataset

        self.split_chosen = loaded_splits
        self.split_train = self.split_chosen['train']
        self.split_val = self.split_chosen['val']
        self.split_test = self.split_chosen['test']

        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1),
            Shift(p=0.5)
        ])

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=None),
            normalize
        ])

        if self.mode == "train":
            self.labels = [self.eeg_dataset[sample_idx][1] for sample_idx in self.split_train]
        elif self.mode == "val":
            self.labels = [self.eeg_dataset[sample_idx][1] for sample_idx in self.split_val]
        elif self.mode == "test":
            self.labels = [self.eeg_dataset[sample_idx][1] for sample_idx in self.split_test]
        else:
            raise ValueError()

        self.labels = torch.tensor(self.labels)
    def __getitem__(self, index):
        """
        Return: (eeg, img), []
            - eeg: Tensor()
            - image: Tensor()
        """
        if self.mode == "train":
            dataset_idx = self.split_train[index]
        elif self.mode == "val":
            dataset_idx = self.split_val[index]
        elif self.mode == "test":
            dataset_idx = self.split_test[index]
        else:
            raise ValueError()
        eeg, label = [self.eeg_dataset[dataset_idx][i] for i in range(0, 2)] #eeg: (1, 62, 800)

        ## If EEG is not downsampled, uncomment below line. For downsample eeg, size is already (128, 128)
        eeg = eeg.squeeze() #(1, 62, 800) to (62, 800)
        eeg = eeg.float()[:, self.time_low:self.time_high] # Cut EEG from (62, 800) => (62, 600)


        # # Add noise to eeg
        # eeg = eeg + torch.randn(eeg.size()) * 0.01
        # Augment eeg using audiomentations (must convert to numpy first)
        if (self.mode == "train"):
            eeg = eeg.numpy()
            ## If EEG is not downsampled, uncomment below line. For downsample eeg, size is already (128, 128)
            eeg = np.array([self.augment(samples=eeg[i], sample_rate=600) for i in range(eeg.shape[0])])
            ##
            # eeg = np.array([self.augment(samples=eeg[i], sample_rate=128) for i in range(eeg.shape[0])])
            eeg = torch.tensor(eeg, dtype=torch.float32)
        
        # Convert eeg to heatmap
        normalized_data = (eeg - eeg.min()) / (eeg.max() - eeg.min())
        # normalized_data = (eeg - eeg.mean()) / (eeg.std()) # Standard scaler
        # normalized_data = F.normalize(input=eeg, p=1, dim=1)
        # grayscale_images = (normalized_data * 255).to(torch.uint8)
        grayscale_images = (normalized_data * 255)
        grayscale_images = grayscale_images.unsqueeze(0).unsqueeze(0) # (1, 1, h, w)
        ## If EEG is not downsampled, uncomment below line.
        eeg_heatmap = F.interpolate(grayscale_images, size=(620, 600), mode='bilinear').squeeze(0).squeeze(0)

        ##
        ## If EEG is downsampled 128Hz, uncomment below line.
        # eeg_heatmap = F.interpolate(grayscale_images, size=(4*128, 128), mode='bilinear')
        #Add edge detection to heatmap image
        eeg_heatmap = eeg_heatmap.to(torch.uint8).numpy()
        # eeg_heatmap = eeg_heatmap.numpy()
        
        eeg_heatmap = cv2.GaussianBlur(eeg_heatmap, (3, 3), 0)     
        edges = cv2.Canny(eeg_heatmap, 50, 100)
        # edges = cv2.adaptiveThreshold(eeg_heatmap, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
        # edges = cv2.convertScaleAbs(edges)
        eeg_heatmap = eeg_heatmap + edges
        
        
        # Can try this to avoid UserWarning
        # eeg_heatmap = eeg_heatmap.clone().detach().requires_grad_(True) 
        eeg_heatmap = torch.tensor(eeg_heatmap, dtype=torch.float32)

        eeg_heatmap = eeg_heatmap.squeeze(0).squeeze(0)
        eeg_heatmap = eeg_heatmap.unsqueeze(0).repeat(3,  1, 1)
        
        eeg_heatmap_resize = self.transform(eeg_heatmap)
        
        return eeg_heatmap_resize, label

    def __len__(self):
        if self.mode == "train":
            return len(self.split_train)
        elif self.mode == "val":
            return len(self.split_val)
        elif self.mode == "test":
            return len(self.split_test)
        else:
            raise ValueError()

class EEG2Image_Dataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """
    def __init__(self, loaded_eeg_heatmaps, loaded_splits, mode="train", transform=None):
        """
        Args:
            img_dir_path: directory path of imagenet images,
            loaded_eeg: eeg dataset loaded from torch.load(),
            loaded_splits: cross-validation splits loaded from torch.load(),
        All arrays and data are returned as torch Tensors
        """
        self.mode = mode
        self.transform = transform
        # self.splits = loaded_splits
        # dataset, classes, img_filenames = [loaded_eeg_heatmaps[k] for k in ['dataset', 'labels', 'images']]
        # self.classes = classes
        # self.img_filenames = img_filenames

        self.eeg_dataset = loaded_eeg_heatmaps

        self.split_chosen = loaded_splits
        self.split_train = self.split_chosen['train']
        self.split_val = self.split_chosen['val']
        self.split_test = self.split_chosen['test']

        if self.mode == "train":
            self.labels = [self.eeg_dataset[sample_idx]['label'] for sample_idx in self.split_train]
        elif self.mode == "val":
            self.labels = [self.eeg_dataset[sample_idx]['label'] for sample_idx in self.split_val]
        elif self.mode == "test":
            self.labels = [self.eeg_dataset[sample_idx]['label'] for sample_idx in self.split_test]
        else:
            raise ValueError()

        self.labels = torch.tensor(self.labels)
    def __getitem__(self, index):
        """
        Return: (eeg, img), []
            - eeg: Tensor()
            - image: Tensor()
        """
        if self.mode == "train":
            dataset_idx = self.split_train[index]
        elif self.mode == "val":
            dataset_idx = self.split_val[index]
        elif self.mode == "test":
            dataset_idx = self.split_test[index]
        else:
            raise ValueError()
        eeg_heatmap,_, label = [self.eeg_dataset[dataset_idx][key] for key in ['eeg', 'image', 'label']]

        eeg_heatmap = eeg_heatmap.unsqueeze(0).repeat(3,  1, 1)

        eeg_heatmap_resize = transforms.Resize((224, 224), antialias=None)(eeg_heatmap)
        return eeg_heatmap_resize, label

    def __len__(self):
        if self.mode == "train":
            return len(self.split_train)
        elif self.mode == "val":
            return len(self.split_val)
        elif self.mode == "test":
            return len(self.split_test)
        else:
            raise ValueError()
        
class EEG2Image_GroupSubject_Dataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """
    def __init__(self, loaded_eeg_heatmaps, loaded_splits, mode="train", transform=None):
        """
        Args:
            img_dir_path: directory path of imagenet images,
            loaded_eeg: eeg dataset loaded from torch.load(),
            loaded_splits: cross-validation splits loaded from torch.load(),
        All arrays and data are returned as torch Tensors
        """
        self.mode = mode
        self.transform = transform
        self.num_subjects = 6
        # self.splits = loaded_splits
        # dataset, classes, img_filenames = [loaded_eeg_heatmaps[k] for k in ['dataset', 'labels', 'images']]
        # self.classes = classes
        # self.img_filenames = img_filenames

        self.eeg_dataset = loaded_eeg_heatmaps

        self.split_chosen = loaded_splits
        self.split_train = self.split_chosen['train']
        self.split_val = self.split_chosen['val']
        self.split_test = self.split_chosen['test']

        # if self.mode == "train":
        #     self.labels = [self.eeg_dataset[sample_idx]['label'] for sample_idx in self.split_train]
        # elif self.mode == "val":
        #     self.labels = [self.eeg_dataset[sample_idx]['label'] for sample_idx in self.split_val]
        # elif self.mode == "test":
        #     self.labels = [self.eeg_dataset[sample_idx]['label'] for sample_idx in self.split_test]
        # else:
        #     raise ValueError()

        # self.labels = torch.tensor(self.labels)
    def __getitem__(self, img_index):
        """
        Return: (eeg, img), []
            - eeg: Tensor()
            - image: Tensor()
        """
        if self.mode == "train":
            indices = self.split_train[img_index]
        elif self.mode == "val":
            indices = self.split_val[img_index]
        elif self.mode == "test":
            indices = self.split_test[img_index]
        else:
            raise ValueError()
        subjects_indices = [self.eeg_dataset[dataset_idx]['subject'] for dataset_idx in indices]
        # Use enumerate to get (index, value) pairs
        indexed_list = list(enumerate(subjects_indices))
        # Sort the indexed list by the values
        sorted_list = sorted(indexed_list, key=lambda x: x[1])
        # Extract the sorted indices
        sorted_indices = [index for index, value in sorted_list]
        # Extract dataset indices by subject sorted indices
        dataset_indices = [indices[index] for index in sorted_indices]

        if (img_index == 0):
            print(f"subjects_indices: {subjects_indices}")
            print(f"sorted_indices: {sorted_indices}")
            print(f"dataset_indices: {dataset_indices}")

        eeg_tensors = [self.eeg_dataset[dataset_idx]['eeg'] for dataset_idx in dataset_indices] # (512, 440)
        labels = [self.eeg_dataset[dataset_idx]['label'] for dataset_idx in dataset_indices]
        eeg_stacked = torch.stack(eeg_tensors, dim=0) # (6, 512, 440)
        # eeg_heatmap,_, label = [self.eeg_dataset[dataset_idx][key] for key in ['eeg', 'image', 'label']]

        eeg_heatmap_resize = transforms.Resize((224, 224), antialias=None)(eeg_stacked)
        return eeg_heatmap_resize, labels[0]

    def __len__(self):
        if self.mode == "train":
            return len(self.split_train)
        elif self.mode == "val":
            return len(self.split_val)
        elif self.mode == "test":
            return len(self.split_test)
        else:
            raise ValueError()

def load_data(eeg_path, eeg_heatmap_path, splits_path, time_low, time_high, splits_by_subject, device, args):
    """
    Args:
        splits_by_subject: bool, whether to use splits by subject or not
    """
    dataset = SEEDIVDataset(io_path=f'/media/mountHDD1/LanxHuyen/SEED_IV/io/',                  
                        root_path='/media/mountHDD1/LanxHuyen/SEED_IV/eeg_raw_data',
                      online_transform=eeg_transforms.Compose([
                          eeg_transforms.ToTensor(),
                          eeg_transforms.To2d()
                      ]),
                      label_transform=eeg_transforms.Select('emotion'))
    # if splits_by_subject:
    #     loaded_eeg_heatmaps = torch.load(eeg_heatmap_path)
    # loaded_splits = torch.load(splits_path)
    
    # Shuffle the indices
    indices = list(range(len(dataset)))
    random.shuffle(list(range(len(dataset))))
    
    # Proportions for train, validation, and test sets (80%, 10%, 10%)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - (train_size + val_size)
    
    # Splitting the indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    loaded_splits = {"train": train_indices, "val": val_indices, "test": test_indices}
    # print(loaded_splits)
    
    if splits_by_subject:
        train_dataset = EEG2Image_GroupSubject_Dataset(loaded_eeg_heatmaps, loaded_splits, mode="train")
        val_dataset = EEG2Image_GroupSubject_Dataset(loaded_eeg_heatmaps, loaded_splits, mode="val")
        test_dataset = EEG2Image_GroupSubject_Dataset(loaded_eeg_heatmaps, loaded_splits, mode="test")
    else:
        train_dataset = EEG2Image_Augment_Dataset(dataset, loaded_splits, time_low, time_high, mode="train")
        val_dataset = EEG2Image_Augment_Dataset(dataset, loaded_splits, time_low, time_high, mode="val")
        test_dataset = EEG2Image_Augment_Dataset(dataset, loaded_splits, time_low, time_high, mode="test")

    kwargs = {'num_workers': 4, 'pin_memory': True} if device else {}
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_dataloader, val_dataloader, test_dataloader
