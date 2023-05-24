from model import load_model
from Encoder.image_encoder import load_image_encoder
from torchinfo import summary

def run(mode="classic", eeg_encoder_name="EEGChannelNet", img_encoder_name="inception_v3"):
    """
    mode: "triplet" | "online_triplet" | "classic"
    """
    # img_encoder = load_image_encoder(img_encoder_name, 1000, feature_extract=True, use_pretrained=True)
    # print(f"Image Encoder: {img_encoder_name}. Input size: {input_size}")
    # print(img_encoder)
    # summary(img_encoder, input_size=(1, 3, 224, 224))

    weight_path = '/home/exx/GithubClonedRepo/EEG-Research/Experiment/ContrastiveLearning/tripletnet_augmented_inceptionv3/model_epoch_50.pth'
    classifier_model = load_model(mode, None, 40, eeg_encoder_name, img_encoder_name)
    summary(classifier_model, input_size=(1, 3, 299, 299))

if __name__ == '__main__':
    run()