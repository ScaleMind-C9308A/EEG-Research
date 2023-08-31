import os
import numpy as np
import random
import argparse
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from data_loader_eeg2image import load_data
from trainer_eeg2image import fit

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from Encoder.image_encoder import load_image_encoder

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def run():
    # Step 0: Setup
    seed_everything(271)
    args = load_config()
    if args.arch != 'test':
        log_path_dir = os.path.join(args.log_path, args.info)
        if not os.path.exists(log_path_dir):
            os.makedirs(log_path_dir)
        logger.add(os.path.join(log_path_dir, 'train.log'))
        logger.info(args)
    is_inception = True if (args.img_encoder == "inception_v3") else False
    # Step 1: Set DataLoaders
    train_dataloader, val_dataloader, test_dataloader = load_data(args.eeg_path,  args.splits_path, args.device,  args)
    # Step 2: Set model
    model = load_image_encoder(args.img_encoder, args.num_classes, args.img_feature_extract, args.use_pretrained)
    model.to(args.device)
    # Step 3: Set loss_fn
    loss_fn = nn.CrossEntropyLoss()
    # Step 4: Set optimizer
    print("Params to learn:")
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    if (args.optim == "Adam"):
        optimizer = optim.Adam(params_to_update, lr=args.lr, weight_decay=args.wd)
    elif (args.optim == "SGD"):
        optimizer = optim.SGD(params_to_update, lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=args.nesterov)

    scheduler = lr_scheduler.StepLR(optimizer, args.lr_step, gamma=0.1, last_epoch=-1)
    #Step 5: Put all to net_trainer()
    fit(train_dataloader, val_dataloader, model, loss_fn, optimizer, scheduler, args.max_epoch, args.device, args.log_interval, log_path_dir, is_inception)



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
    parser = argparse.ArgumentParser(description='Online Triplet Training of EEG and image')
    parser.add_argument('--dataset',
                        help='Dataset name.')
    parser.add_argument('--eeg-path',
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