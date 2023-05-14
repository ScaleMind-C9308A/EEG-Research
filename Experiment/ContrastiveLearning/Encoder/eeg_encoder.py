# from EEG_Encoder.LSTM import classifier_LSTM
# from EEG_Encoder.LSTM_per_channel import classifier_LSTM_per_channel
# from EEG_Encoder.Stacked_BiLSTM import classifier_Stacked_BiLSTM
# from EEG_Encoder.CNN import classifier_CNN
# from EEG_Encoder.EEGNet import classifier_EEGNet
# from EEG_Encoder.SyncNet import classifier_SyncNet
from Encoder.EEGChannelNet import EEGChannelNet

# def load_eeg_encoder(
#              n_classes,
#              classes,
#              encoder,
#              length, # 512
#              channel, # 96
#              min_CNN,
#              kind):
#         if encoder=="EEGChannelNet":
#             net = EEGChannelNet()
    
#         # print("DONE: CREATE TORCH CLASSIFIER")
#         # print(net)
#         return net

def load_eeg_encoder(encoder):
    if encoder=="EEGChannelNet":
        net = EEGChannelNet()

    # print("DONE: CREATE EEG ENCODER")
    # print(net)
    return net