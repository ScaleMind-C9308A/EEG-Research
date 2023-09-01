import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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

def load_data(eeg_heatmap_path, splits_path, device, args):
    """
    Args:
        is_inception: True | False
    """
    loaded_eeg_heatmaps = torch.load(eeg_heatmap_path)
    loaded_splits = torch.load(splits_path)
    
    train_dataset = EEG2Image_Dataset(loaded_eeg_heatmaps, loaded_splits, mode="train")
    val_dataset = EEG2Image_Dataset(loaded_eeg_heatmaps, loaded_splits, mode="val")
    test_dataset = EEG2Image_Dataset(loaded_eeg_heatmaps, loaded_splits, mode="test")

    kwargs = {'num_workers': 4, 'pin_memory': True} if device else {}
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_dataloader, val_dataloader, test_dataloader