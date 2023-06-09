import torch
import torch.nn as nn
import torch.nn.functional as F
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
    is_inception = True if (args.img_encoder == "inception_v3") else False 
    # Define your model architecture
    train_dataloader, val_dataloader, test_dataloader = load_data(args.eeg_path, args.img_path, args.splits_path, args.time_low, args.time_high, args.device, mode=args.classifier_mode, img_encoder=args.img_encoder)
    # Step 2: Set model
    model = load_model(mode=args.classifier_mode, weight_path=args.weight_path, embedding_dim=args.embedding_size, num_classes=args.num_classes, eeg_encoder_name=args.eeg_encoder, img_encoder_name=args.img_encoder)
    # print(model)
    model.to(args.device)

    # # Identify the layer from which you want to extract the embeddings
    # feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    # # Set the feature extractor to evaluation mode
    # feature_extractor.eval()
    model.eval()
    # Extract embeddings from the validation set
    eeg_features = None
    img_features = None
    img_pos_features = None
    img_neg_features = None
    labels = np.array([])


   # Loop over the validation embeddings and concatenate them into a single tensor
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_dataloader):
            # if (batch_idx >= 5): # only take first 5 batches => 64*5=320 samples
            #    break
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if args.device:
                data = tuple(d.to(args.device) for d in data)
                # data = torch.Tensor(data)
                # print(type(data))
                if target is not None:
                    target = target.to(args.device)
            labels = np.concatenate((labels, target.detach().cpu().numpy()))
        
            # # Extract embeddings from the desired layer
            # embeddings = feature_extractor(data)
            if (args.classifier_mode == "triplet"):
                eeg, img1, img2 = data    
                eeg_feature = model.get_eeg_embedding(eeg).detach().cpu().numpy()
                img_pos_feature = model.get_img_embedding(img1).detach().cpu().numpy()
                img_neg_feature = model.get_img_embedding(img2).detach().cpu().numpy()
                if eeg_features is not None:
                    eeg_features = np.concatenate((eeg_features, eeg_feature))
                else:
                    eeg_features = eeg_feature
                if img_pos_features is not None:
                    img_pos_features = np.concatenate((img_pos_features, img_pos_feature))
                else:
                    img_pos_features = img_pos_feature
                if img_neg_features is not None:
                    img_neg_features = np.concatenate((img_neg_features, img_neg_feature))
                else:
                    img_neg_features = img_neg_feature
            elif (args.classifier_mode == "online_triplet"):
                eeg, img = data
                eeg_feature = model.get_eeg_embedding(eeg).detach().cpu().numpy()
                img_feature = model.get_img_embedding(img).detach().cpu().numpy()
                if eeg_features is not None:
                    eeg_features = np.concatenate((eeg_features, eeg_feature))
                else:
                    eeg_features = eeg_feature
                if img_features is not None:
                    img_features = np.concatenate((img_features, img_feature))
                else:
                    img_features = img_feature
            elif (args.classifier_mode == "classic_eeg"):
                eeg, img = data
                eeg_feature = model.get_eeg_embedding(eeg).detach().cpu().numpy()
                if eeg_features is not None:
                    eeg_features = np.concatenate((eeg_features, eeg_feature))
                else:
                    eeg_features = eeg_feature
                

        # validation_embeddings.append(embeddings)

        # validation_embeddings = torch.cat(validation_embeddings, dim=0)

        # Convert the embeddings to a numpy array
        # embeddings_np = validation_embeddings.numpy()

        # print(f"Image feature size: {eeg_features.shape}")
        # print(eeg_features)
        # Perform dimensionality reduction with t-SNE
        eeg_tsne = TSNE(n_components=2, random_state=42).fit_transform(eeg_features)
        # img_tsne = TSNE(n_components=2, random_state=42).fit_transform(img_features)
        # img_pos_tsne = TSNE(n_components=2, random_state=42).fit_transform(img_pos_features)
        # img_neg_tsne = TSNE(n_components=2, random_state=42).fit_transform(img_neg_features)
        # print(f"tsne eeg size: {eeg_tsne.shape}")
        # print(eeg_tsne)

        
        visualize_tsne(eeg_tsne, labels, log_path_dir, info="EEG")
        # visualize_tsne(img_tsne, labels, log_path_dir, info = "Image")
        # visualize_tsne(img_pos_tsne, labels, log_path_dir, info="Image_positive")
        # visualize_tsne(img_neg_tsne, labels, log_path_dir, info="Image_negative")

def generate_random_colors(n):
    """
    Generates a list of n distinct random colors.

    Args:
        n: The number of colors to generate.

    Returns:
        A list of n distinct random colors.
    
    """

    colors = {}
    for i in range(n):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors[i] = (b, g, r)

    return colors


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def visualize_tsne_points(tx, ty, labels, dict_class_to_color, log_path_dir, info):
    #initialize save_fig path
    save_fig_path = os.path.join(log_path_dir, f"{info}_tsne.png")
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(f"Labels length: {len(labels)}")
    # print("dict class to color")
    # print(dict_class_to_color)

    # for every class, we'll add a scatter plot separately
    for label in dict_class_to_color:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([dict_class_to_color[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')
    plt.title(f"{info} tsne plot")

    # finally, show the plot
    plt.savefig(save_fig_path)

def visualize_tsne(tsne, labels, log_path_dir, info, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    dict_class_to_color = generate_random_colors(40)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels, dict_class_to_color, log_path_dir, info)

    # # visualize the plot: samples as images
    # visualize_tsne_images(tx, ty, images, labels, plot_size=plot_size, max_image_size=max_image_size)

# Perform further analysis or evaluation with the obtained embeddings
def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='Argparser')
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
                        help='Name of eeg encoder')
    
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
