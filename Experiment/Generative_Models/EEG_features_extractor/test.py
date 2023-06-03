import torch
import argparse
import numpy as np

def test():
    args = load_config()
    loaded_eeg = torch.load(args.eeg_path)
    loaded_splits = torch.load(args.splits_path)['splits'][0]
    eeg_dataset, classes, img_filenames = [loaded_eeg[k] for k in ['dataset', 'labels', 'images']]
    # Create labels
    labels_train = [eeg_dataset[sample_idx]['label'] for sample_idx in loaded_splits['train']]
    labels_val = [eeg_dataset[sample_idx]['label'] for sample_idx in loaded_splits['val']]
    labels_test = [eeg_dataset[sample_idx]['label'] for sample_idx in loaded_splits['test']]

    img_train = np.array([eeg_dataset[sample_idx]['image'] for sample_idx in loaded_splits['train']])
    img_val = np.array([eeg_dataset[sample_idx]['image'] for sample_idx in loaded_splits['val']])
    img_test = np.array([eeg_dataset[sample_idx]['image'] for sample_idx in loaded_splits['test']])

    train_img_to_indices = {img: np.where(img_train == img)[0]
                                    for img in set(img_train)}
    val_img_to_indices = {img: np.where(img_val == img)[0]
                                    for img in set(img_val)}
    test_img_to_indices = {img: np.where(img_test == img)[0]
                                    for img in set(img_test)}
    for key in train_img_to_indices:
        if key <10:
            indices = train_img_to_indices[key]
            print(f"Number of indices of img {key} is: {len(indices)}")

def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='Online Triplet Training of EEG and image')

    parser.add_argument('--eeg-path', default=None, 
                        help='Path of eeg data')
    parser.add_argument('--img-path', default=None, 
                        help='Path of image dataset directory')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--splits-path',
                        help='Path of splits dataset')
    
    
    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    return args

if __name__ == "__main__":
    test()