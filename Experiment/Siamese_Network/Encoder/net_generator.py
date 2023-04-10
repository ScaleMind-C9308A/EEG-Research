# from EEG_Encoder.LSTM import classifier_LSTM
# from EEG_Encoder.LSTM_per_channel import classifier_LSTM_per_channel
# from EEG_Encoder.Stacked_BiLSTM import classifier_Stacked_BiLSTM
# from EEG_Encoder.CNN import classifier_CNN
# from EEG_Encoder.EEGNet import classifier_EEGNet
# from EEG_Encoder.SyncNet import classifier_SyncNet
from Encoder.EEGChannelNet import classifier_EEGChannelNet

def Classifier(
             n_classes,
             classes,
             classifier,
             GPUindex,
             length, # 512
             channel, # 96
             min_CNN,
             kind):
        if classifier=="EEGChannelNet":
            if length<min_CNN:
                return
            if kind=="from-scratch":
                output_size = len(classes)
            if kind=="incremental":
                output_size = n_classes
            if kind=="no-model-file":
                output_size = len(classes)
            net = classifier_EEGChannelNet().cuda(GPUindex)
    
        print("DONE: CREATE TORCH CLASSIFIER")
        print(net)
        nonclasses = [i for i in range(output_size) if i not in classes]
        return net, nonclasses