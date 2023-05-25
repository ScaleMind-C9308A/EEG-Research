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
        model: "inception_v3" | "resnet50"
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
                transforms.ToTensor(),  # Convert the image to a PyTorch tensor
                normalize
        ])


def load_data(eeg_path, img_path, splits_path, eeg_time_low, eeg_time_high, device, mode="triple", is_inception=False):
    """
    mode: "triple" | "online_triplet",
    img_encoder: "inception_v3" | "resnet50"
    """
    loaded_eeg = torch.load(eeg_path)
    loaded_splits = torch.load(splits_path)['splits']
    train_transform = img_transform(is_inception, mode="train")
    val_transform = img_transform(is_inception, mode="val")
    if (mode== "triple"):
        train_dataset = EEGDataset_Triple(img_path, loaded_eeg, loaded_splits, eeg_time_low,eeg_time_high, mode="train", transform=train_transform)
        val_dataset = EEGDataset_Triple(img_path, loaded_eeg, loaded_splits, eeg_time_low,eeg_time_high,mode="val", transform=val_transform)
        test_dataset = EEGDataset_Triple(img_path, loaded_eeg, loaded_splits, eeg_time_low,eeg_time_high,mode="test", transform=val_transform)
    elif (mode=="online_triplet"):
        train_dataset = EEGDataset(img_path, loaded_eeg, loaded_splits,eeg_time_low,eeg_time_high, mode="train", transform=train_transform)
        val_dataset = EEGDataset(img_path, loaded_eeg, loaded_splits,eeg_time_low,eeg_time_high, mode="val", transform=val_transform)
        test_dataset = EEGDataset(img_path, loaded_eeg, loaded_splits,eeg_time_low,eeg_time_high, mode="test", transform=val_transform)
    train_batch_sampler = BalancedBatchSampler(train_dataset.labels, n_classes=8, n_samples=8)
    val_batch_sampler = BalancedBatchSampler(val_dataset.labels, n_classes=8, n_samples=8)
    test_batch_sampler = BalancedBatchSampler(test_dataset.labels, n_classes=8, n_samples=8)

    kwargs = {'num_workers': 1, 'pin_memory': True} if device else {}
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, **kwargs)
    test_dataloader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)
    return train_dataloader, val_dataloader, test_dataloader

class EEGDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """
    def __init__(self, img_dir_path, loaded_eeg, loaded_splits, time_low, time_high, mode="train", transform=None):
        """
        Args:
            img_dir_path: directory path of imagenet images,
            loaded_eeg: eeg dataset loaded from torch.load(),
            loaded_splits: cross-validation splits loaded from torch.load(),
            time_low (int): lowest time value
            time_high (int): highest time value
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
        self.time_low = time_low
        self.time_high = time_high
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
        eeg, img_positive_idx, label = [self.eeg_dataset[dataset_idx][key] for key in ['eeg', 'image', 'label']]
        eeg = eeg.float()[:, self.time_low:self.time_high]
        img_positive_filename, img_positive_classname = self.img_filenames[img_positive_idx], self.classes[label]
        chosen_crop_pos = np.random.choice(range(10))
        img_pos_crop = f"{img_positive_filename}_crop_{chosen_crop_pos}"
        img_positive = Image.open(os.path.join(self.img_dir_path, img_pos_crop+'.jpeg' )).convert('RGB')

        if self.transform is not None:
            img_positive = self.transform(img_positive)
        return (eeg, img_positive), label

    def __len__(self):
        if self.mode == "train":
            return len(self.split_train)
        elif self.mode == "val":
            return len(self.split_val)
        elif self.mode == "test":
            return len(self.split_test)
        else:
            raise ValueError()
        
class EEGDataset_Triple(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """
    def __init__(self, img_dir_path, loaded_eeg, loaded_splits, time_low, time_high, mode="train", transform=None):
        """
        Args:
            img_dir_path: directory path of imagenet images,
            loaded_eeg: eeg dataset loaded from torch.load(),
            loaded_splits: cross-validation splits loaded from torch.load(),

        """
        # self.opt = opt
        # # Load EEG signals
        # loaded_eeg = torch.load(eeg_signals_path)
        # # Load splits file
        # loaded_splits = torch.load(block_splits_path)
        self.mode = mode
        self.transform = transform
        self.img_dir_path = img_dir_path
        self.splits = loaded_splits
        dataset, classes, img_filenames = [loaded_eeg[k] for k in ['dataset', 'labels', 'images']]
        self.classes = classes
        self.img_filenames = img_filenames

        self.eeg_dataset = dataset
        self.time_low = time_low
        self.time_high = time_high
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
        self.labels_set = set(self.labels.numpy())
        """self.label_to_indices: Map each label to its corresponding samples in the dataset"""
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                    for label in self.labels_set}
        # random_state = np.random.RandomState(29)

        # triplets = [[i,
        #              random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
        #              random_state.choice(self.label_to_indices[
        #                                      np.random.choice(
        #                                          list(self.labels_set - set([self.test_labels[i].item()]))
        #                                      )
        #                                  ])
        #              ]
        #             for i in range(len(self.test_data))]
        # self.test_triplets = triplets

    def __getitem__(self, index):
        """
        ds = EEGDataset()
        ds[i] will return by what we define __getitem__ magic methods
        """
        if self.mode == "train":
            dataset_idx = self.split_train[index]
        elif self.mode == "val":
            dataset_idx = self.split_val[index]
        elif self.mode == "test":
            dataset_idx = self.split_test[index]
        else:
            raise ValueError()
        eeg, img_positive_idx, img_positive_label = [self.eeg_dataset[dataset_idx][key] for key in ['eeg', 'image', 'label']]
        eeg = eeg.float()[:, self.time_low:self.time_high]
        # We don't need to sample neg image from different labels,
        # we only need to sample neg image that are different from pos_image
        # img_negative_label = np.random.choice(list(self.labels_set - set([img_positive_label])))
        # sample_negative_idx = np.random.choice(self.label_to_indices[img_negative_label])
        img_negative_idx = np.random.choice(np.delete(np.array(range(len(self.img_filenames))), img_positive_idx))
        # print(f"Len img filenames: {len(self.img_filenames)}")
        # print(f"img to indices: {self.label_to_indices}")
        # print(f"img_negative_label: {img_negative_label}")
        # print(f"img_pos_idx: {img_positive_idx}")
        # print(f"img_neg_idx: {img_negative_idx}")
        img_positive_filename = self.img_filenames[img_positive_idx]
        img_negative_filename = self.img_filenames[img_negative_idx]
        chosen_crop_pos = np.random.choice(range(10))
        chosen_crop_neg = np.random.choice(range(10))
        img_pos_crop = f"{img_positive_filename}_crop_{chosen_crop_pos}"
        img_neg_crop = f"{img_negative_filename}_crop_{chosen_crop_neg}"
        img_positive = Image.open(os.path.join(self.img_dir_path, img_pos_crop+'.jpeg' )).convert('RGB')
        img_negative = Image.open(os.path.join(self.img_dir_path, img_neg_crop+'.jpeg' )).convert('RGB')
        # else:
        #     img1 = self.test_data[self.test_triplets[index][0]]
        #     img2 = self.test_data[self.test_triplets[index][1]]
        #     img3 = self.test_data[self.test_triplets[index][2]]

        # img1 = Image.fromarray(img1.numpy(), mode='L')
        # img2 = Image.fromarray(img2.numpy(), mode='L')
        # img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img_positive = self.transform(img_positive)
            img_negative = self.transform(img_negative)
        return (eeg, img_positive, img_negative), []

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
