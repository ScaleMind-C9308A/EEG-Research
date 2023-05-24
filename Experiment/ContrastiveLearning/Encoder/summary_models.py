from image_encoder import load_image_encoder
from torchinfo import summary

def run(eeg_encoder_name="EEGChannelNet", img_encoder_name="resnet50"):
    """
    model: "triple_net" | "embedding_net"
    """
    # eeg_encoder = load_eeg_encoder(eeg_encoder)
    img_encoder = load_image_encoder(img_encoder_name, 1000, pretrained=True)
    print(f"Image Encoder: {img_encoder_name}")
    print(img_encoder)
    summary(img_encoder, input_size=(1, 3, 299, 299))

if __name__ == '__main__':
    run()