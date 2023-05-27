# from EEG_Encoder.LSTM import classifier_LSTM
# from EEG_Encoder.LSTM_per_channel import classifier_LSTM_per_channel
# from EEG_Encoder.Stacked_BiLSTM import classifier_Stacked_BiLSTM
# from EEG_Encoder.CNN import classifier_CNN
# from EEG_Encoder.EEGNet import classifier_EEGNet
# from EEG_Encoder.SyncNet import classifier_SyncNet
import torch
import torch.nn as nn
from Encoder.EEGChannelNet import EEGChannelNet_Encoder

def load_eeg_encoder(encoder_name, output_dim):
    if encoder_name=="EEGChannelNet":
        net = EEGChannelNet_Encoder(embedding_size=output_dim)

    # print("DONE: CREATE EEG ENCODER")
    # print(net)
    return net