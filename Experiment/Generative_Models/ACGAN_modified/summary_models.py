# from image_encoder import load_image_encoder
from model import load_model
from torchinfo import summary
import torch

def run(eeg_encoder_name="EEGChannelNet", img_encoder_name="inception_v3"):
    """
    model: "triple_net" | "embedding_net"
    """
    # eeg_encoder = load_eeg_encoder(eeg_encoder)
    weight_path = "/home/exx/GithubClonedRepo/EEG-Research/Experiment/ContrastiveLearning/tripletnet_augmented_inceptionv3/model_epoch_50.pth"
    model = load_model("triplet", weight_path=weight_path, num_classes=40, eeg_encoder_name=eeg_encoder_name, img_encoder_name=img_encoder_name)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    # print(f"Image Encoder: {img_encoder_name}")
    print(model)
    # summary(feature_extractor, input_size=(1, 3, 299, 299))

if __name__ == '__main__':
    run()