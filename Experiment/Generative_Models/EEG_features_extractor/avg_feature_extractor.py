import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

from Encoder.eeg_encoder import load_eeg_encoder
from Encoder.image_encoder import load_image_encoder, load_image_encoder_triplet

def avg_feature_extract():
    """
    Return:
        eeg_features: Tensor([num_classes, eeg_embedding_size])
    """
    args = load_config()
    #Load model for inference
    model = EEGClassificationNet('EEGChannelNet', args.embedding_size, 40)
    model.load_state_dict(torch.load(args.weight_path))
    model.to(args.device)
    model.eval()
    #Load eeg data
    loaded_eeg = torch.load(args.eeg_path)
    loaded_splits = torch.load(args.splits_path)['splits'][0]
    eeg_dataset, classes, img_filenames = [loaded_eeg[k] for k in ['dataset', 'labels', 'images']]
    # Create labels
    labels_train = np.array([eeg_dataset[sample_idx]['label'] for sample_idx in loaded_splits['train']])
    labels_val = np.array([eeg_dataset[sample_idx]['label'] for sample_idx in loaded_splits['val']])
    labels_test = np.array([eeg_dataset[sample_idx]['label'] for sample_idx in loaded_splits['test']])
    train_label_to_indices = {label: np.where(labels_train == label)[0]
                                    for label in set(labels_train)}
    val_label_to_indices = {label: np.where(labels_val == label)[0]
                                    for label in set(labels_val)}
    test_label_to_indices = {label: np.where(labels_test == label)[0]
                                    for label in set(labels_test)}
    # Calculate avg eeg embeddings on each class

class EEGClassificationNet(nn.Module):
    def __init__(self, backbone_name, embedding_dim, num_classes):
        super().__init__()
        self.backbone = load_eeg_encoder(backbone_name, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_classes)
            # # For nn.CrossEntropyLoss() => The input is expected to contain the unnormalized logits for each class
            # nn.LogSoftmax(dim=1)
        )
    def forward(self, eeg):
        return self.backbone(eeg)
    def get_eeg_embedding(self, eeg):
        return self.backbone(eeg)
    
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
        eeg, img_positive_idx, label = [self.eeg_dataset[dataset_idx][key] for key in ['eeg', 'image', 'label']]
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

def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='Online Triplet Training of EEG and image')

    # From argparse document: The bool() function is not recommended as a type converter. All it does is convert 
    # empty strings to False and non-empty strings to True
    parser.add_argument('--embedding-size', default=1000, type=int,
                        help="Embedding size for training")
    parser.add_argument('--weight-path', default=None, 
                        help='Path of pretrained weight of the model')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    
    
    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    return args