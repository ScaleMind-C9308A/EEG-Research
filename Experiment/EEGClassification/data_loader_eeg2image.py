import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np

class EEG2Image_Augment_Dataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """
    def __init__(self, loaded_eeg, loaded_splits,time_low, time_high, mode="train", transform=None):
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

        self.time_low = time_low
        self.time_high = time_high
        dataset, classes, img_filenames = [loaded_eeg[k] for k in ['dataset', 'labels', 'images']]
        self.classes = classes
        self.img_filenames = img_filenames

        self.eeg_dataset = dataset

        self.split_chosen = loaded_splits
        self.split_train = self.split_chosen['train']
        self.split_val = self.split_chosen['val']
        self.split_test = self.split_chosen['test']

        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            Shift(p=0.5),
        ])

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
        eeg,_, label = [self.eeg_dataset[dataset_idx][key] for key in ['eeg', 'image', 'label']]
        eeg = eeg.float()[:, self.time_low:self.time_high] # (channels, time_steps) => (128, 440)
        # # Add noise to eeg
        # eeg = eeg + torch.randn(eeg.size()) * 0.01
        # Augment eeg using audiomentations (must convert to numpy first)
        eeg = eeg.numpy()
        eeg = np.array([self.augment(samples=eeg[i], sample_rate=440) for i in range(eeg.shape[0])])
        eeg = torch.tensor(eeg, dtype=torch.float32)
        # Convert eeg to heatmap
        normalized_data = (eeg - eeg.min()) / (eeg.max() - eeg.min())
        grayscale_images = (normalized_data * 255).to(torch.uint8)
        grayscale_images = grayscale_images.unsqueeze(0).unsqueeze(0) # (1, 1, h, w)
        # eeg_heatmap = F.interpolate(grayscale_images, size=(512, 440), mode='nearest', align_corners=False)
        eeg_heatmap = F.interpolate(grayscale_images, size=(512, 440), mode='nearest')
        eeg_heatmap = eeg_heatmap.squeeze(0).squeeze(0)
        # Can try this to avoid UserWarning
        # eeg_heatmap = eeg_heatmap.clone().detach().requires_grad_(True) 
        eeg_heatmap = torch.tensor(eeg_heatmap, dtype=torch.float32)

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
    loaded_eeg = torch.load(eeg_path)
    loaded_eeg_heatmaps = torch.load(eeg_heatmap_path)
    loaded_splits = torch.load(splits_path)
    
    if splits_by_subject:
        train_dataset = EEG2Image_GroupSubject_Dataset(loaded_eeg_heatmaps, loaded_splits, mode="train")
        val_dataset = EEG2Image_GroupSubject_Dataset(loaded_eeg_heatmaps, loaded_splits, mode="val")
        test_dataset = EEG2Image_GroupSubject_Dataset(loaded_eeg_heatmaps, loaded_splits, mode="test")
    else:
        train_dataset = EEG2Image_Augment_Dataset(loaded_eeg, loaded_splits, time_low, time_high, mode="train")
        val_dataset = EEG2Image_Augment_Dataset(loaded_eeg, loaded_splits, time_low, time_high, mode="val")
        test_dataset = EEG2Image_Augment_Dataset(loaded_eeg, loaded_splits, time_low, time_high, mode="test")

    kwargs = {'num_workers': 4, 'pin_memory': True} if device else {}
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_dataloader, val_dataloader, test_dataloader
