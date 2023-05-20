import os
import torch
from loguru import logger
import argparse
import random
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from PIL import Image

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    np.random.seed(seed)
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def augmentation(): 

    seed_everything(271)
    args = load_config()
    if args.arch != 'test':
        log_path_dir = os.path.join(args.log_path, args.info)
        if not os.path.exists(log_path_dir):
            os.makedirs(log_path_dir)
        logger.add(os.path.join(log_path_dir, 'train.log'))
        logger.info(args)

    # Define the data augmentation transformations
    transform = transforms.Compose([
        transforms.Resize(299),  # Resize the image to 1.1 times the expected input size
        transforms.TenCrop(299),  # Extract ten crops
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        # transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops])),
    ])

    # Load image paths
    image_folder = args.img_path
    image_paths = os.listdir(args.img_path)

    # Create a directory to save the augmented images
    output_dir = args.img_path_aug
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the dataset and save the augmented images
    for image_path in image_paths:
        image = Image.open(os.path.join(image_folder, image_path)).convert("RGB")  # Load image
        augmented_images = transform(image) # Apply transformations and add batch dimension

        # Save each augmented image
        for i, augmented_image in enumerate(augmented_images):
            image_name = image_path.split('.')[0]
            image_filename = f'{image_name}_crop_{i}.jpeg'
            # print(image_filename)
            image_path = os.path.join(output_dir, image_filename)
            augmented_image = augmented_image.mul(255).byte().numpy().transpose(1, 2, 0)  # Convert tensor to PIL Image format
            augmented_image = Image.fromarray(augmented_image)
            augmented_image.save(image_path)
            
            

        print(f'Saved 10 augmented images for image {image_path}.')

    print('Data augmentation and saving complete.')
def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='Inference Inception V3 _ ImageNet')
    parser.add_argument('--dataset',
                        help='Dataset name.')
    parser.add_argument('--img-path',
                        help='Path of image dataset')
    parser.add_argument('--img-path-aug',
                        help='Path of image augmentation')
    parser.add_argument('--log-path', 
                        help="Directory path to save log files during training")
    parser.add_argument('--info', default='Trivial',
                        help='Train info')
    parser.add_argument('--img-encoder', default="inception_v3", type=str,
                        help='inception_v3 | resnet50')
    parser.add_argument('--arch', default='train',
                        help='Net arch')
    parser.add_argument('--save_ckpt', default='checkpoints/',
                        help='result_save')
    
    
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
    augmentation()