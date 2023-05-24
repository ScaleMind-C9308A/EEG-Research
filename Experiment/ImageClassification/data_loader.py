from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import os

def img_transform(is_inception, mode="train"):
    """
    Training images transform.

    Args
        is_inception: True | False
        mode: "train" | "val"
    Returns
        transform(torchvision.transforms): transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_size = (299,299) if is_inception else (224,224)
    if (mode == "train"):
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),                         
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif (mode=="val"):
        return transforms.Compose([
                transforms.Resize(img_size),  # Resize the image to 299x299 pixels
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),  # Convert the image to a PyTorch tensor
                normalize
        ])


def load_data(eeg_path, img_path, splits_path, device, is_inception, args):
    """
    Args:
        is_inception: True | False
    """
    loaded_eeg = torch.load(eeg_path)
    loaded_splits = torch.load(splits_path)['splits']
    train_transform = img_transform(is_inception, mode="train")
    val_transform = img_transform(is_inception, mode="val")
    
    train_dataset = ImageDataset(img_path, loaded_eeg, loaded_splits, mode="train", transform=train_transform)
    val_dataset = ImageDataset(img_path, loaded_eeg, loaded_splits, mode="val", transform=val_transform)
    test_dataset = ImageDataset(img_path, loaded_eeg, loaded_splits, mode="test", transform=val_transform)

    # train_batch_sampler = BalancedBatchSampler(train_dataset.labels, n_classes=8, n_samples=8)
    # val_batch_sampler = BalancedBatchSampler(val_dataset.labels, n_classes=8, n_samples=8)
    # test_batch_sampler = BalancedBatchSampler(test_dataset.labels, n_classes=8, n_samples=8)

    kwargs = {'num_workers': 4, 'pin_memory': True} if device else {}
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_dataloader, val_dataloader, test_dataloader

class ImageDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """
    def __init__(self, img_dir_path, loaded_eeg, loaded_splits, mode="train", transform=None):
        """
        Args:
            img_dir_path: directory path of imagenet images,
            loaded_eeg: eeg dataset loaded from torch.load(),
            loaded_splits: cross-validation splits loaded from torch.load(),
        All arrays and data are returned as torch Tensors
        """
        self.mode = mode
        self.transform = transform
        self.img_dir_path = img_dir_path
        self.splits = loaded_splits
        dataset, classes, img_filenames = [loaded_eeg[k] for k in ['dataset', 'labels', 'images']]
        self.classes = classes
        self.img_filenames = img_filenames

        self.eeg_dataset = dataset
        """We use only split 0, no cross-validation"""
        self.split_chosen = loaded_splits[0]
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
        _, img_idx, label = [self.eeg_dataset[dataset_idx][key] for key in ['eeg', 'image', 'label']]
        img_filename, img_classname = self.img_filenames[img_idx], self.classes[label]
        img = Image.open(os.path.join(self.img_dir_path, img_filename+'.JPEG' )).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        if self.mode == "train":
            return len(self.split_train)
        elif self.mode == "val":
            return len(self.split_val)
        elif self.mode == "test":
            return len(self.split_test)
        else:
            raise ValueError()
        

    
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - samples n_classes and within these classes samples n_samples.
    Each iter will return a batch of indices of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels) # num of samples in dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
