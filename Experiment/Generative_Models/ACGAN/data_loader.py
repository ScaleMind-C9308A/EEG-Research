from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import os

def img_transform(mode="train", img_size=128):
    """
    Training images transform.

    Args
        model: "inception_v3" | "resnet50"
        mode: "train" | "val"
    Returns
        transform(torchvision.transforms): transform
    """
    #GAN Normalization
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if (mode == "train"):
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),                         
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    elif (mode=="val"):
        return transforms.Compose([
            transforms.Resize(img_size),  # Resize the image to 299x299 pixels
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

def make_dataset(dir, classes, class_to_idx):
    images = []
    labels = []
    dir = os.path.expanduser(dir)
    for class_name in classes:
        class_dir = os.path.join(dir, class_name)
        if os.path.exists(class_dir):
            for fname in sorted(os.listdir(class_dir)):
                if is_image_file(fname):
                    path = os.path.join(class_dir, fname)
                    item = (path, class_to_idx[class_name])
                    images.append(item)
                    labels.append(class_to_idx[class_name])
    return images, labels

def extract_classes(classes, chosen_classes_idx=None):
    """
    classes: list of class names
    chosen_classes_idx: list of chosen classes' index
    Return:
        chosen_classes: list of chosen classes' name
        class_to_idx: dict of {class_name: idx}
    """
    if chosen_classes_idx is not None:
        chosen_classes = [classes[i] for i in chosen_classes_idx]
    else:
        chosen_classes = classes
    class_to_idx = {chosen_classes[i]: i for i in range(len(chosen_classes))}
    return chosen_classes, class_to_idx

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

    def __init__(self, root, classes, chosen_classes_idx=None, transform=None, target_transform=None):
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.chosen_classes_idx = chosen_classes_idx
        chosen_classes, class_to_idx = extract_classes(classes, self.chosen_classes_idx)
        imgs, labels = make_dataset(root,chosen_classes, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs #(num_imgs, )
        self.labels = labels #(num_imgs,)
        self.classes = chosen_classes #(40, )
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
        
class GANDataset(Dataset):
    """
    Each sample (__getitem__) of dataset returns:
        Label (Target)
        An image
        Average EEG feature vector over all images of the selected class and over all subjects
    """
    def __init__(self, img_dir_path, loaded_eeg, loaded_splits, label_to_eeg_embeddings, mode="train", transform=None):
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
        self.label_to_eeg_embeddings = label_to_eeg_embeddings

        self.eeg_dataset = dataset
        # self.time_low = time_low
        # self.time_high = time_high
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
        Return:
            Label (Target)
            An image
            Average EEG feature vector over all images of the selected class and over all subjects
        """
        if self.mode == "train":
            dataset_idx = self.split_train[index]
        elif self.mode == "val":
            dataset_idx = self.split_val[index]
        elif self.mode == "test":
            dataset_idx = self.split_test[index]
        else:
            raise ValueError()
        _, img_idx, label_pos = [self.eeg_dataset[dataset_idx][key] for key in ['eeg', 'image', 'label']]
        # label_neg = np.random.choice(self.labels_set - label_pos)
        avg_eeg_embedding_pos = self.label_to_eeg_embeddings[label_pos]
        # avg_eeg_embedding_neg = self.label_to_eeg_embeddings[label_neg]
        
        img_filename = self.img_filenames[img_idx]
        img = Image.open(os.path.join(self.img_dir_path, img_filename+'.JPEG' )).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        return (img, avg_eeg_embedding_pos), label_pos

    def __len__(self):
        if self.mode == "train":
            return len(self.split_train)
        elif self.mode == "val":
            return len(self.split_val)
        elif self.mode == "test":
            return len(self.split_test)
        else:
            raise ValueError()
    

def load_data(eeg_path, img_w_eeg_path, img_no_eeg_path, eeg_embeddings_path, splits_path, args):
    """
    Return:
        train_loader_stage1: each __getitem__ returns (real_img)
        train_loader_stage2: each __getitem__ returns (real_img, eeg)
        val_loader: each __getitem__ returns (real_img, eeg)
    """
    loaded_eeg = torch.load(eeg_path)
    loaded_splits = torch.load(splits_path)['splits']
    label_to_eeg_embeddings = torch.load(eeg_embeddings_path)
    _, classes, _ = [loaded_eeg[k] for k in ['dataset', 'labels', 'images']]
    train_transform = img_transform("train", args.img_size)
    val_transform = img_transform("val", args.img_size)

    chosen_classes_idx = range(args.start_class, args.end_class+1)

    train_ds_stage1 = ImageFolder(img_no_eeg_path, classes, chosen_classes_idx, train_transform)
    train_ds_stage2 = GANDataset(img_w_eeg_path, loaded_eeg, loaded_splits, label_to_eeg_embeddings, mode="train", transform=train_transform)
    val_ds_stage2 = GANDataset(img_w_eeg_path, loaded_eeg, loaded_splits, label_to_eeg_embeddings, mode="val", transform=val_transform)

    options_train = {
        'num_workers': args.num_workers, 
        'pin_memory': True,
        'batch_size': args.batch_size,
        'shuffle': True
        }
    options_val = {
        'num_workers': args.num_workers, 
        'pin_memory': True,
        'batch_size': 64,
        'shuffle': False
        }
    train_loader_stage1 = DataLoader(train_ds_stage1, **options_train)
    train_loader_stage2 = DataLoader(train_ds_stage2, **options_train)
    val_loader = DataLoader(val_ds_stage2, **options_val)
    return train_loader_stage1, train_loader_stage2, val_loader