import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os

from Encoder.eeg_encoder import load_eeg_encoder
from Encoder.image_encoder import load_image_encoder, load_image_encoder_triplet

def avg_feature_extract():
    """
    Return:
        eeg_features: <Dict> {<label>: <avg eeg embedding>}
    """
    args = load_config()
    #Load model for inference
    model = EEGClassificationNet('EEGChannelNet', args.embedding_size, 40)
    model.load_state_dict(torch.load(args.weight_path))
    # model.to(args.device)
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
    label_to_eeg_embeddings = {}
    with torch.no_grad():
        for sample_class in train_label_to_indices:
            indices = train_label_to_indices[sample_class]
            eeg_embeddings = torch.empty((len(indices), args.embedding_size))
            for i, idx in enumerate(indices):
                eeg, img, label = [eeg_dataset[idx][key] for key in ['eeg', 'image', 'label']]
                eeg = eeg.float()[:, args.time_low:args.time_high]
                eeg = eeg.unsqueeze(0)
                # eeg.to(args.device)
                
                # print(f"EEG Size: {eeg.size()}")
                eeg_embedding = model.get_eeg_embedding(eeg)
                eeg_embeddings[i] = eeg_embedding
            average_eeg_embedding = torch.mean(eeg_embeddings, dim=0)
            # print(average_eeg_embedding)
            label_to_eeg_embeddings[sample_class] = average_eeg_embedding
        torch.save(label_to_eeg_embeddings, os.path.join(args.save_path, f"{args.info}.pth"))    

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

    ### SPECIFIC TO THIS SCRIPT
    parser.add_argument('--save-path', type=str,
                        help='Directory to save.')
    parser.add_argument('--info', type=str,
                        help='Save file name')
    parser.add_argument('--embedding-size', type=int, default=1000,
                        help='EEG Embedding size')
    parser.add_argument('--weight-path', default=None, 
                        help='Path of pretrained weight of the model')
    ###########################
    parser.add_argument('--dataset', type=str, default="CVPR2017",
                        help='Dataset name.')
    parser.add_argument('--eeg-path',
                        help='Path of eeg dataset')
    parser.add_argument('--time-low', type=float, default=20,
                        help='Lowest time value of eeg segment')
    parser.add_argument('--time-high', type=float, default=460,
                        help='highest time value of eeg segment')
    parser.add_argument('--img-path',
                        help='Path of image dataset')
    parser.add_argument('--splits-path',
                        help='Path of splits dataset')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    
    
    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    return args

if __name__ == "__main__":
    avg_feature_extract()