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
from Encoder.LSTM_Dropout import classifier_LSTM_Dropout
from Encoder.Stacked_BiLSTM_Dropout import classifier_Stacked_BiLSTM_Dropout
from Encoder.Stacked_BiLSTM_Combined import Encoder_Stacked_BiLSTM_Combined
from Encoder.EEGWaveNet import EEGWaveNet_Encoder   
from Encoder.SyncNet import classifier_SyncNet

def load_eeg_encoder(encoder, embedding_dim, device):
    if encoder=="EEGChannelNet":
        net = EEGChannelNet_Encoder(embedding_size=embedding_dim)
    elif encoder=="EEGChannelNet_Modified":
        net = EEGChannelNet_Encoder_Mod(embedding_size=embedding_dim)
    elif encoder=="EEGNet":
        net = classifier_EEGNet(embedding_size=embedding_dim)
    elif encoder=="EEGWaveNet":
        net = EEGWaveNet_Encoder(n_chans=128)
    elif encoder=="SyncNet":
        net = classifier_SyncNet(embedding_size=embedding_dim)
    elif encoder=="Stacked_BiLSTM":
        net = classifier_Stacked_BiLSTM(embedding_size=embedding_dim, device=device)
    elif encoder=="LSTM":
        net=classifier_LSTM(embedding_size=embedding_dim, device=device)
    elif encoder=="LSTM_Dropout":
        net=classifier_LSTM_Dropout(embedding_size=embedding_dim, device=device)
    elif encoder=="Stacked_BiLSTM_Dropout":
        net=classifier_Stacked_BiLSTM_Dropout(embedding_size=embedding_dim, device=device)
    elif encoder=="Stacked_BiLSTM_Combined":
        net=Encoder_Stacked_BiLSTM_Combined(embedding_size=embedding_dim, device=device)

    # print("DONE: CREATE EEG ENCODER")
    # print(net)
    return net