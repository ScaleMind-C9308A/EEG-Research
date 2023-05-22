import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# Added pkgs for configurations
import argparse
from loguru import logger
import os
import numpy as np
import random
# Import modules
from data_loader_aug import load_data
from model import load_model
from losses import TripletLoss
from trainer import fit
from metrics import AverageNonzeroTripletsMetric

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def run_triplet():   
    """
    Detailed steps in run:
    Step 0: Setup and preprocessing
        - Define options (can use ArgParser)
            + eeg_dir_path, img_dir_path
            + lr, 
    Step 1: Set Dataloaders (data_loader.py)
        - From load_data(eeg_path, image_path, ...): 
            return train_dataloader, val_dataloader and test_dataloader
    Step 2: Set model (model.py)
    Step 3: Set loss_fn (losses.py)
    Step 4: Set optimizer (Adam/SGD)
    Step 5: Put all to net_trainer()
    """
    # Step 0: Setup
    seed_everything(271)
    args = load_config()
    if args.arch != 'test':
        log_path_dir = os.path.join(args.log_path, args.info)
        if not os.path.exists(log_path_dir):
            os.makedirs(log_path_dir)
        logger.add(os.path.join(log_path_dir, 'train.log'))
        logger.info(args)

    # Step 1: Set DataLoaders
    train_dataloader, val_dataloader, test_dataloader = load_data(args.eeg_path, args.img_path, args.splits_path, args.time_low, args.time_high, args.device, mode='triple', img_encoder=args.img_encoder)
    # Step 2: Set model
    model = load_model(model="triplet_net", eeg_encoder=args.eeg_encoder, img_encoder=args.img_encoder)
    model.to(args.device)
    # Step 3: Set loss_fn
    margin = 0.
    loss_fn = TripletLoss(margin)
    # Step 4: Set optimizer
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    #Step 5: Put all to net_trainer()
    fit(train_dataloader, val_dataloader, model, loss_fn, optimizer, scheduler, args.max_epoch, args.device, args.log_interval)

def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='Triplet Training of EEG and image')
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
                        help='inception_v3 | resnet50')
    
    # Model training configurations
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size.(default: 128)')
    parser.add_argument('--lr', default=5e-3, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='Weight Decay.(default: 1e-4)')
    parser.add_argument('--optim', default='SGD', type=str,
                        help='Optimizer')
    # parser.add_argument('--max-iter', default=40, type=int,
    #                     help='Number of iterations.(default: 40)')
    parser.add_argument('--max-epoch', default=100, type=int,
                        help='Number of epochs.(default: 30)')
    parser.add_argument('--log-interval', default=10, type=int,
                        help='Log interval during training.(default: 10)')
    
    # GPU training settings
    parser.add_argument('--num-workers', default=2, type=int,
                        help='Number of loading data threads.(default: 4)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    
    # Hyperparameters
    parser.add_argument('--gamma', default=200, type=float,
                        help='Hyper-parameter.(default: 200)')
    parser.add_argument('--arch', default='train',
                        help='Net arch')
    parser.add_argument('--save_ckpt', default='checkpoints/',
                        help='result_save')
    parser.add_argument('--lr-step', default='40', type=str,
                        help='lr decrease step.(default: 40)')
    parser.add_argument('--pretrain', action='store_true',
                        help='Using image net pretrain')
    parser.add_argument('--momen', default=0.9, type=float,
                        help='Hyper-parameter.(default: 0.9)')
    parser.add_argument('--nesterov', action='store_true',
                        help='Using SGD nesterov')
    parser.add_argument('--num-classes', default=40, type=int,
                        help='Number of classes.(default: 40)')
    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    return args

if __name__ == '__main__':
    run_triplet()
