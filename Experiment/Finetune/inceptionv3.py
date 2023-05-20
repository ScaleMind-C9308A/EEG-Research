import torch
from torchvision import models, transforms, datasets
import os
import numpy as np
from loguru import logger
import argparse
import random
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
def fine_tune():
    seed_everything(271)
    args = load_config()
    if args.arch != 'test':
        log_path_dir = os.path.join(args.log_path, args.info)
        if not os.path.exists(log_path_dir):
            os.makedirs(log_path_dir)
        logger.add(os.path.join(log_path_dir, 'train.log'))
        logger.info(args)

# Load pre-trained Inception v3 model
    model = models.inception_v3(pretrained=True)

    # # Load ImageNet class labels
    # labels = os.listdir(args.img_path)[img].split('\\')[-1].split('_')[0] 

    # Set up data transformations
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Load image paths
    image_paths = args.img_path

    # Perform inference
    model.eval()  # Set the model to evaluation mode

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")  # Load image
        image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
        image = image.to(args.device)  # Move image to device

        with torch.no_grad():
            outputs = model(image)  # Forward pass
            percents = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            top5_vals, top5_inds = percents.topk(5)
            # _, predicted_idx = torch.max(outputs, 1)  # Get predicted class index
            # predicted_label = labels[predicted_idx.item()]  # Map index to class label
        print(f"Top5 index: {top5_inds} - Top5 vals: {top5_vals}")
        # print(f"Image: {image_path} - Predicted label: {predicted_label}")
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
    fine_tune()