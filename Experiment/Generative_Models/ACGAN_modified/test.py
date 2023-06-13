import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Added pkgs for configurations
import argparse
from loguru import logger
import os
import numpy as np
import random
from data_loader import load_data
from model import load_model
from trainer_GAN import trainer_GAN
import matplotlib.pyplot as plt
import random

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
    # Define your model architecture
    train_loader_stage1, train_loader_stage2, val_loader = load_data(args.eeg_path, args.img_w_eeg_path, args.img_no_eeg_path, args.eeg_embedding_path, args.splits_path, args)
    # Step 2: Set model
    # Initialize the netG and netD
    if args.pretrained_netG != None and args.pretrained_netD != None:
        logger.info("Use pretrained netG and netD for stage1 training")
        is_pretrained_stage1 = True
    else:
        is_pretrained_stage1 = False
    netG, netD = load_model(args.num_classes, args.latent_dim, args.eeg_dim, is_pretrained_stage1, args.pretrained_netG, args.pretrained_netD)
    netG = netG.to(args.device)
    netD = netD.to(args.device)

    # Step 3: Set loss_fn/criterion
    # loss functions
    dis_criterion = nn.BCELoss()
    aux_criterion = nn.NLLLoss()
    # Step 4: Set optimizer
    if (args.optim == "Adam"):
        optimizer_G = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))    

    # scheduler = lr_scheduler.StepLR(optimizer, args.lr_step, gamma=0.1, last_epoch=-1)
    #Step 5: Put all to net_trainer()
    trainer_GAN(train_loader_stage1, train_loader_stage2, val_loader, netG, netD, dis_criterion, aux_criterion, optimizer_G, optimizer_D, is_pretrained_stage1, None, log_path_dir, args)


    
def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='Argparser')
    ### Specific to DCGAN
    # From argparse document: The bool() function is not recommended as a type converter. All it does is convert 
    # empty strings to False and non-empty strings to True

    parser.add_argument('--img-size', default=64, type=int,
                        help="Latent z (Noise) vector dimension")
    parser.add_argument('--eeg-dim', default=128, type=int,
                        help="EEG (Condition) vector dimension")
    parser.add_argument('--latent-dim', default=100, type=int,
                        help="Latent z (Noise) vector dimension")
    parser.add_argument('--weight-path', default=None, 
                        help='Path of pretrained weight of the model')
    parser.add_argument('--img-w-eeg-path', default=None, 
                        help='Path of image dataset that has recorded EEG data')
    parser.add_argument('--img-no-eeg-path', default=None, 
                        help='Path of image dataset that has NO recorded EEG data')
    parser.add_argument('--eeg-embedding-path', default=None, 
                        help='Path of extracted average EEG embeddings')
    parser.add_argument('--pretrained-netG', default=None, 
                        help='If pretrained: path of pretrained generator')
    parser.add_argument('--pretrained-netD', default=None, 
                        help='If pretrained: path of pretrained discriminator')
    parser.add_argument('--num-epochs-stage1', default=100, type=int,
                        help="Number of epochs trained at stage1 of GAN")
    parser.add_argument('--num-epochs-stage2', default=50, type=int,
                        help="Number of epochs trained at stage2 of GAN")
    parser.add_argument('--start-class', default=0, type=int,
                        help='Start class for training')
    parser.add_argument('--end-class', default=9, type=int,
                        help='End class for training')
    ##################################
    parser.add_argument('--dataset',
                        help='Dataset name.')
    parser.add_argument('--eeg-path',
                        help='Path of eeg dataset')
    parser.add_argument('--splits-path',
                        help='Path of splits dataset')
    parser.add_argument('--log-path', 
                        help="Directory path to save log files during training")
    parser.add_argument('--info', default='Trivial',
                        help='Train info')
    
    # Model training configurations
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size.(default: 128)')
    parser.add_argument('--log-interval', default=10, type=int,
                        help='Log interval during training.(default: 10)')
    parser.add_argument('--arch', default='train',
                        help='Net arch')
    parser.add_argument('--save_ckpt', default='checkpoints/',
                        help='result_save')
    parser.add_argument('--num-classes', default=40, type=int,
                        help='Number of classes.(default: 40)')
    parser.add_argument(
    '--G_param', type=str, default='SN',
    help='Parameterization style to use for G, spectral norm (SN) or SVD (SVD)'
          ' or None (default: %(default)s)')
    parser.add_argument(
        '--D_param', type=str, default='SN',
        help='Parameterization style to use for D, spectral norm (SN) or SVD (SVD)'
            ' or None (default: %(default)s)')    
    parser.add_argument(
        '--G_ch', type=int, default=64,
        help='Channel multiplier for G (default: %(default)s)')
    parser.add_argument(
        '--D_ch', type=int, default=64,
        help='Channel multiplier for D (default: %(default)s)')
    parser.add_argument(
        '--G_depth', type=int, default=1,
        help='Number of resblocks per stage in G? (default: %(default)s)')
    parser.add_argument(
        '--D_depth', type=int, default=1,
        help='Number of resblocks per stage in D? (default: %(default)s)')
    parser.add_argument(
        '--D_thin', action='store_false', dest='D_wide', default=True,
        help='Use the SN-GAN channel pattern for D? (default: %(default)s)')
    parser.add_argument(
        '--G_shared', action='store_true', default=False,
        help='Use shared embeddings in G? (default: %(default)s)')
    parser.add_argument(
        '--shared_dim', type=int, default=0,
        help='G''s shared embedding dimensionality; if 0, will be equal to dim_z. '
            '(default: %(default)s)')
    parser.add_argument(
        '--dim_z', type=int, default=128,
        help='Noise dimensionality: %(default)s)')
    parser.add_argument(
        '--z_var', type=float, default=1.0,
        help='Noise variance: %(default)s)')    
    parser.add_argument(
        '--hier', action='store_true', default=False,
        help='Use hierarchical z in G? (default: %(default)s)')
    parser.add_argument(
        '--cross_replica', action='store_true', default=False,
        help='Cross_replica batchnorm in G?(default: %(default)s)')
    parser.add_argument(
        '--mybn', action='store_true', default=False,
        help='Use my batchnorm (which supports standing stats?) %(default)s)')
    parser.add_argument(
        '--G_nl', type=str, default='relu',
        help='Activation function for G (default: %(default)s)')
    parser.add_argument(
        '--D_nl', type=str, default='relu',
        help='Activation function for D (default: %(default)s)')
    parser.add_argument(
        '--G_attn', type=str, default='64',
        help='What resolutions to use attention on for G (underscore separated) '
            '(default: %(default)s)')
    parser.add_argument(
        '--D_attn', type=str, default='64',
        help='What resolutions to use attention on for D (underscore separated) '
            '(default: %(default)s)')
    parser.add_argument(
        '--norm_style', type=str, default='bn',
        help='Normalizer style for G, one of bn [batchnorm], in [instancenorm], '
            'ln [layernorm], gn [groupnorm] (default: %(default)s)')
    
    # GPU training settings
    parser.add_argument('--num-workers', default=4, type=int,
                        help='Number of loading data threads.(default: 4)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')    
    
    # Training Optim Settings
    parser.add_argument('--lr', default=2e-4, type=float,
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
        ### Optimizer stuff ###
    parser.add_argument(
        '--G_lr', type=float, default=5e-5,
        help='Learning rate to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_lr', type=float, default=2e-4,
        help='Learning rate to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B1', type=float, default=0.0,
        help='Beta1 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B1', type=float, default=0.0,
        help='Beta1 to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B2', type=float, default=0.999,
        help='Beta2 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B2', type=float, default=0.999,
        help='Beta2 to use for Discriminator (default: %(default)s)')
    
    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    return args

if __name__ == '__main__':
    run()
