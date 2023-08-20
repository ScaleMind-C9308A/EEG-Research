# from EEG_Encoder.LSTM import classifier_LSTM
# from EEG_Encoder.LSTM_per_channel import classifier_LSTM_per_channel
# from EEG_Encoder.Stacked_BiLSTM import classifier_Stacked_BiLSTM
# from EEG_Encoder.CNN import classifier_CNN
# from EEG_Encoder.EEGNet import classifier_EEGNet
# from EEG_Encoder.SyncNet import classifier_SyncNet
import torch
import torch.nn as nn
from Encoder.EEGChannelNet import EEGChannelNet_Encoder
from Encoder.EEGChannelNet_modified import EEGChannelNet_Encoder_Mod
from Encoder.EEGNet import classifier_EEGNet
from Encoder.Stacked_BiLSTM import classifier_Stacked_BiLSTM
from Encoder.LSTM import classifier_LSTM

def load_eeg_encoder(encoder, embedding_dim):
    if encoder=="EEGChannelNet":
        net = EEGChannelNet_Encoder(embedding_size=embedding_dim)
    elif encoder=="EEGChannelNet_Modified":
        net = EEGChannelNet_Encoder_Mod(embedding_size=embedding_dim)
    elif encoder=="EEGNet":
        net = classifier_EEGNet(embedding_size=embedding_dim)
    elif encoder=="Stacked_BiLSTM":
        net = classifier_Stacked_BiLSTM(embedding_size=embedding_dim)
    elif encoder=="LSTM":
        net=classifier_LSTM(embedding_size=embedding_dim)

    # print("DONE: CREATE EEG ENCODER")
    # print(net)
    return net