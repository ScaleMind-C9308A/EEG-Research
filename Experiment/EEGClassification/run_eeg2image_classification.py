import os
import numpy as np
import argparse
from loguru import logger

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from Encoder.image_encoder import load_image_encoder

def run():
    args = load_config()

    log_path_dir = os.path.join(args.log_path, args.info)
    if not os.path.exists(log_path_dir):
        os.makedirs(log_path_dir)
    logger.add(os.path.join(log_path_dir, 'train.log'))
    logger.info(args)
    # Step 1: Prepare your CNN model (e.g., ResNet18)
    cnn_model = load_image_encoder(args.img_encoder, args.embedding_size, True, args.pretrained)
    cnn_model = cnn_model.eval()  # Set the model to evaluation mode

    # Step 2: Extract features using the CNN model
    train_dataloader, val_dataloader, test_dataloader = load_data(args.eeg_path, args.splits_path, args.device, args)
    cnn_features, targets = cnn_feature_extractor(train_dataloader, cnn_model, args.device)

    # # Flatten the features if needed
    # flattened_features = cnn_features.view(cnn_features.size(0), -1).numpy()

    # Step 3: Split data and train SVM classifier
    X_train, X_test, y_train, y_test = train_test_split(cnn_features, targets, test_size=0.2, random_state=42)

    # Create an SVM classifier with RBF kernel
    svm_classifier = SVC(kernel='rbf', C=1.0, gamma='auto')  # Adjust hyperparameters as needed

    # Train the SVM classifier
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = svm_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.2f}")

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
        self.splits = loaded_splits
        dataset, classes, img_filenames = [loaded_eeg_heatmaps[k] for k in ['dataset', 'labels', 'images']]
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
        eeg,_, label = [self.eeg_dataset[dataset_idx][key] for key in ['eeg', 'image', 'label']]

        return eeg, label

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
    loaded_splits = torch.load(splits_path)['splits']
    
    train_dataset = EEG2Image_Dataset(loaded_eeg_heatmaps, loaded_splits, mode="train")
    val_dataset = EEG2Image_Dataset(loaded_eeg_heatmaps, loaded_splits, mode="val")
    test_dataset = EEG2Image_Dataset(loaded_eeg_heatmaps, loaded_splits, mode="test")

    # train_batch_sampler = BalancedBatchSampler(train_dataset.labels, n_classes=8, n_samples=8)
    # val_batch_sampler = BalancedBatchSampler(val_dataset.labels, n_classes=8, n_samples=8)
    # test_batch_sampler = BalancedBatchSampler(test_dataset.labels, n_classes=8, n_samples=8)

    kwargs = {'num_workers': 4, 'pin_memory': True} if device else {}
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_dataloader, val_dataloader, test_dataloader

def cnn_feature_extractor(data_loader, model, device):
    input_features = []  # Collect model outputs for SVM input
    targets = []          # Corresponding SVM targets
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, targets) in enumerate(data_loader):
            # print(f"Batch {batch_idx}, batch_size: {len(target)}")
            # print(f"EEG size: {data[0].size()}")
            # print(f"Image size: {data[1].size()}")
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            
            input_features.append(outputs.cpu().detach().numpy())  # Collect outputs for SVM
            targets.append(targets.cpu().detach().numpy())  # Collect targets for SVM

        input_features = np.vstack(input_features)  # Stack collected features into a matrix
        targets = np.concatenate(targets)            # Concatenate collected targets

    return input_features, targets

def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='EEG-to-Image Based Model')
    ### Specific to Contrastive Learning
    # From argparse document: The bool() function is not recommended as a type converter. All it does is convert 
    # empty strings to False and non-empty strings to True
    parser.add_argument('--img-feature-extract', default=0, type=int,
                        help='(1|0: Option to turn on feature extraction of image encoder')
    parser.add_argument('--embedding-size', default=1000, type=int,
                        help="Embedding size for training")
    parser.add_argument('--classifier-mode', default='classic', type=str,
                        help='classic | triplet | online_triplet')
    parser.add_argument('--weight-path', default=None, 
                        help='Path of pretrained weight of the model')
    ##################################
    parser.add_argument('--dataset',
                        help='Dataset name.')
    parser.add_argument('--eeg-heatmap-path',
                        help='Path of eeg dataset')
    parser.add_argument('--img-path',
                        help='Path of image dataset')
    parser.add_argument('--splits-path',
                        help='Path of splits dataset')
    parser.add_argument('--log-path', 
                        help="Directory path to save log files during training")
    parser.add_argument('--info', default='Trivial',
                        help='Train info')
    parser.add_argument('--img-encoder', default="inception_v3", type=str,
                        help='inception_v3 | resnet50')

    
    # Model training configurations
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size.(default: 128)')
    parser.add_argument('--max-epoch', default=100, type=int,
                        help='Number of epochs.(default: 30)')

    parser.add_argument('--arch', default='train',
                        help='Net arch')
    parser.add_argument('--save_ckpt', default='checkpoints/',
                        help='result_save')
    parser.add_argument('--num-classes', default=40, type=int,
                        help='Number of classes.(default: 40)')
    
    # GPU training settings
    parser.add_argument('--num-workers', default=2, type=int,
                        help='Number of loading data threads.(default: 4)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')    
    

    
    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    return args


if __name__ == '__main__':
    run()

