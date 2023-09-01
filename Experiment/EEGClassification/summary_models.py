from model import load_model
from Encoder.image_encoder import load_image_encoder
from Encoder.eeg_encoder import load_eeg_encoder
import argparse
from loguru import logger
from torchinfo import summary
import os
import torch

def run(mode="classic", eeg_encoder_name="LSTM", img_encoder_name="inception_v3"):
    """
    mode: "triplet" | "online_triplet" | "classic"
    """
    # img_encoder = load_image_encoder(img_encoder_name, 1000, feature_extract=True, use_pretrained=True)
    # print(f"Image Encoder: {img_encoder_name}. Input size: {input_size}")
    # print(img_encoder)
    # summary(img_encoder, input_size=(1, 3, 224, 224))

    weight_path = '/home/exx/GithubClonedRepo/EEG-Research/Experiment/ContrastiveLearning/tripletnet_augmented_inceptionv3/model_epoch_50.pth'
    
    args = load_config()
    
    logger.add(os.path.join(args.log_path, f"{args.info}.log"))
    logger.info(args)
    classifier_model = load_image_encoder(args.img_encoder, args.embedding_size, True, args.use_pretrained)
    # classifier_model = load_model(args.classifier_mode, args.weight_path, args.num_classes, args.eeg_encoder, 
                                #   args.img_encoder, args.embedding_size,False, args.device)
    classifier_model.to(args.device)
    logger.info(summary(classifier_model, input_size=(args.batch_size, 128, 440)))

def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='Online Triplet Training of EEG and image')
    ### Specific to Contrastive Learning
    # From argparse document: The bool() function is not recommended as a type converter. All it does is convert 
    # empty strings to False and non-empty strings to True
    parser.add_argument('--embedding-size', default=128, type=int,
                        help="Embedding size for training")
    parser.add_argument('--classifier-mode', default='classic', type=str,
                        help='classic | triplet | online_triplet')
    parser.add_argument('--weight-path', default=None, 
                        help='Path of pretrained weight of the model')
    ##################################
    
    parser.add_argument('--log-path', 
                        help="Directory path to save log files during training")
    parser.add_argument('--info', default='Trivial',
                        help='Train info')
    parser.add_argument('--img-encoder', default="inception_v3", type=str,
                        help='inception_v3 | resnet50')
    parser.add_argument('--eeg-encoder', default="EEGChannelNet", type=str,
                        help='inception_v3 | resnet50')
    
    # Model training configurations
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-classes', default=40, type=int,
                        help='Number of classes.(default: 40)')
    
    # GPU training settings
    
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