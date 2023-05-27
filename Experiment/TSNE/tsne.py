import torch
import torch.nn as nn
from model import load_model
# Added pkgs for configurations
import argparse
from loguru import logger
import os
import numpy as np
import random
from data_loader import load_data
from model import load_model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def run():  
    seed_everything(271)
    args = load_config()
    if args.arch != 'test':
        log_path_dir = os.path.join(args.log_path, args.info)
        if not os.path.exists(log_path_dir):
            os.makedirs(log_path_dir)
        logger.add(os.path.join(log_path_dir, 'train.log'))
        logger.info(args)
    is_inception = True if (args.img_encoder == "inception_v3") else False 
    # Define your model architecture
    train_dataloader, val_dataloader, test_dataloader = load_data(args.eeg_path, args.img_path, args.splits_path, args.time_low, args.time_high, args.device, mode='triple', img_encoder=args.img_encoder)
    # Step 2: Set model
    model = load_model(mode=args.classifier_mode, weight_path=args.weight_path, num_classes=args.num_classes, eeg_encoder_name=args.eeg_encoder, img_encoder_name=args.img_encoder)
    model.to(args.device)

    # Identify the layer from which you want to extract the embeddings
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    # Set the feature extractor to evaluation mode
    feature_extractor.eval()
    # Extract embeddings from the validation set
    validation_embeddings = []
   # Loop over the validation embeddings and concatenate them into a single tensor
    for batch_idx, (data, targets) in enumerate(val_dataloader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if args.device:
            data = tuple(d.to(args.device) for d in data)
            if target is not None:
                target = target.to(args.device)

        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        with torch.no_grad():
            # Extract embeddings from the desired layer
            embeddings = feature_extractor(outputs)

        validation_embeddings.append(embeddings)

    validation_embeddings = torch.cat(validation_embeddings, dim=0)

    # Convert the embeddings to a numpy array
    embeddings_np = validation_embeddings.numpy()

    # Perform dimensionality reduction with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings_np)

    # Extract the labels for plotting
    labels = targets.numpy()
    # Plot the t-SNE visualization
    save_fig_tnse = os.path.join(log_path_dir, 'plot_tsne.png')
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title("t-SNE Visualization of Embeddings")
    plt.savefig(save_fig_tnse)

    # Perform further analysis or evaluation with the obtained embeddings
def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='Online Triplet Training of EEG and image')
    parser.add_argument('--dataset',
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
    parser.add_argument('--log-path', 
                        help="Directory path to save log files during training")
    parser.add_argument('--info', default='Trivial',
                        help='Train info')
    parser.add_argument('--img-encoder', default="inception_v3", type=str,
                        help='inception_v3 | resnet50')
    parser.add_argument('--eeg-encoder', default="EEGChannelNet", type=str,
                        help='eeg encoder')
    parser.add_argument('--classifier-mode', default='classic', type=str,
                        help='classic | triplet | online_triplet')
    parser.add_argument('--weight-path', default=None, 
                        help='Path of pretrained weight of the model')
    
    # Model training configurations
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size.(default: 128)')
    parser.add_argument('--max-epoch', default=100, type=int,
                        help='Number of epochs.(default: 30)')
    parser.add_argument('--log-interval', default=10, type=int,
                        help='Log interval during training.(default: 10)')
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
    
    # Training Optim Settings
    parser.add_argument('--lr', default=2.5e-4, type=float,
                    help='Learning rate.(default: 2.5e-4)')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='Weight Decay.(default: 1e-4)')
    parser.add_argument('--optim', default='Adam', type=str,
                        help='Optimizer')
    parser.add_argument('--gamma', default=200, type=float,
                        help='Hyper-parameter.(default: 200)')
    parser.add_argument('--lr-step', default=40, type=int,
                        help='lr decrease step.(default: 40)')
    parser.add_argument('--pretrain', action='store_true',
                        help='Using image net pretrain')
    parser.add_argument('--momen', default=0.9, type=float,
                        help='Hyper-parameter.(default: 0.9)')
    parser.add_argument('--nesterov', action='store_true',
                        help='Using SGD nesterov')
    
    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    return args

if __name__ == '__main__':
    run()
