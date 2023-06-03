from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import os

def img_transform(mode="train"):
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
    # img_size = (299,299) if (model=="inception_v3") else (224,224)
    if (mode == "train"):
        return transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.RandomCrop((64, 64)),                         
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif (mode=="val"):
        return transforms.Compose([
            transforms.Resize((64, 64)),  # Resize the image to 299x299 pixels
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            normalize
        ])

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def make_dataset(dir, class_to_idx):
    images = []
    labels = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        if target not in class_to_idx:
            continue
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    labels.append(class_to_idx[target])

    return images

def find_classes(dir, classes_idx=None):
    """
    class_to_idx: dict that map each class name to its index (0...40)
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    if classes_idx is not None:
        assert type(classes_idx) == tuple
        start, end = classes_idx
        classes = classes[start:end]
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class ImageFolder(Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, classes, transform=None, target_transform=None,
                 classes_idx=None):
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes_idx = classes_idx
        # classes, class_to_idx = find_classes(root, self.classes_idx)
        imgs, labels = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs #(num_imgs, )
        self.labels = labels #(num_imgs,)
        self.classes = classes #(40, )
        self.class_to_idx = class_to_idx #<Dict> {class1: idx1,...}
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
        
class GANDatasetStage2(Dataset):
    """
    Each sample (__getitem__) of dataset returns:
        Label (Target)
        An image
        Average EEG feature vector over all images of the selected class and over all subjects
    """
    def __init__(self, img_dir_path, loaded_eeg, loaded_splits, class_to_eeg_embeddings, time_low, time_high, mode="train", transform=None):
        """
        Args:
            img_dir_path: directory path of imagenet images,
            loaded_eeg: eeg dataset loaded from torch.load(),
            loaded_splits: cross-validation splits loaded from torch.load(),
            class_to_eeg_embeddings: <Dict> {<class>: <torch tensor: eeg_embedding>}

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
        self.labels_set = set(self.labels.numpy())
        """self.label_to_indices: Map each label to its corresponding samples in the dataset"""
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                    for label in self.labels_set}

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
        eeg, img_idx, label = [self.eeg_dataset[dataset_idx][key] for key in ['eeg', 'image', 'label']]
        eeg = eeg.float()[:, self.time_low:self.time_high]
        
        img_positive_filename = self.img_filenames[img_positive_idx]
        img_positive = Image.open(os.path.join(self.img_dir_path, img_positive_filename+'.JPEG' )).convert('RGB')
        
        if self.transform is not None:
            img_positive = self.transform(img_positive)
            img_negative = self.transform(img_negative)
        return (eeg, img_positive, img_negative), label

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

def load_data(eeg_path, img_path, splits_path, eeg_time_low, eeg_time_high, device, mode="triple", img_encoder="inception_v3"):
    """
    mode: "triple" | "online_triplet",
    img_encoder: "inception_v3" | "resnet50"

    Return:
        train_loader_stage1: each __getitem__ returns (real_img)
        train_loader_stage2: each __getitem__ returns (real_img, eeg)
        val_loader: each __getitem__ returns (real_img, eeg)
    """
    loaded_eeg = torch.load(eeg_path)
    loaded_splits = torch.load(splits_path)['splits']
    _, classes, _ = [loaded_eeg[k] for k in ['dataset', 'labels', 'images']]
    train_transform = img_transform(mode="train")
    val_transform = img_transform(mode="val")
    if (mode== "triplet"):
        train_dataset = EEGDataset_Triple(img_path, loaded_eeg, loaded_splits, eeg_time_low,eeg_time_high, mode="train", transform=train_transform)
        val_dataset = EEGDataset_Triple(img_path, loaded_eeg, loaded_splits, eeg_time_low,eeg_time_high,mode="val", transform=val_transform)
        test_dataset = EEGDataset_Triple(img_path, loaded_eeg, loaded_splits, eeg_time_low,eeg_time_high,mode="test", transform=val_transform)
    elif (mode=="online_triplet" or mode=="classic_eeg"):
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