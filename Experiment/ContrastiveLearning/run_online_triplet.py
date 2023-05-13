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
from data_loader import load_data
from model import load_model
from losses import OnlineTripletLoss
from trainer import fit
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from metrics import AverageNonzeroTripletsMetric

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def run_online_triplet():   
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
        if not os.path.exists(args.log_path + args.info):
            os.makedirs(args.log_path + args.info)
        logger.add(args.log_path + args.info  + '/' + 'train.log')
        logger.info(args)

    # Step 1: Set DataLoaders
    train_dataloader, val_dataloader, test_dataloader = load_data(args.eeg_path, args.img_path, args.splits_path, args.device, mode='online_triplet')
    # Step 2: Set model
    model = load_model(model="embedding_net")
    # Step 3: Set loss_fn
    margin = 0.
    loss_fn = OnlineTripletLoss(margin, args.device, RandomNegativeTripletSelector(margin))
    # Step 4: Set optimizer
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    #Step 5: Put all to net_trainer()/fit()
    fit(train_dataloader, val_dataloader, model, loss_fn, optimizer, scheduler, args.max_epoch, args.device, args.log_interval, [AverageNonzeroTripletsMetric])
    # net_trainer(train_dataloader, val_dataloader, model, loss_fn, optimizer, scheduler, args.max_epoch, args.device, args.log_interval, metrics=[AverageNonzeroTripletsMetric()])

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
    parser.add_argument('--eeg_path',
                        help='Path of eeg dataset')
    parser.add_argument('--img_path',
                        help='Path of image dataset')
    parser.add_argument('--splits_path',
                        help='Path of splits dataset')
    parser.add_argument('--log-path', 
                        help="Directory path to save log files during training")
    parser.add_argument('--info', default='Trivial',
                        help='Train info')
    # Model training configurations
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size.(default: 128)')
    parser.add_argument('--lr', default=2.5e-4, type=float,
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
    run_online_triplet()
