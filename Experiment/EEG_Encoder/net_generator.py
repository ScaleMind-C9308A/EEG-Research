from EEG_Encoder.LSTM import classifier_LSTM
from EEG_Encoder.LSTM_per_channel import classifier_LSTM_per_channel
from EEG_Encoder.Stacked_BiLSTM import classifier_Stacked_BiLSTM
from EEG_Encoder.CNN import classifier_CNN
from EEG_Encoder.EEGNet import classifier_EEGNet
from EEG_Encoder.SyncNet import classifier_SyncNet
from EEG_Encoder.EEGChannelNet import classifier_EEGChannelNet

def Classifier(
             n_classes,
             classes,
             classifier,
             GPUindex,
             length, # 512
             channel, # 96
             min_CNN,
             kind):
    if classifier=="LSTM":
        if kind=="from-scratch":
            output_size = 128
        if kind=="incremental":
            output_size = 128
        if kind=="no-model-file":
            output_size = 128
        net = classifier_LSTM(
            True,
            input_size = channel,
            lstm_layers = 1,
            lstm_size = 128,
            output1_size = 128,
            output2_size = None,
            GPUindex = GPUindex)
    elif classifier=="LSTM1":
        if kind=="from-scratch":
            output_size = 128
        if kind=="incremental":
            output_size = 128
        if kind=="no-model-file":
            output_size = 128
        net = classifier_LSTM(
            False,
            input_size = channel,
            lstm_layers = 1,
            lstm_size = 128,
            output1_size = 128,
            output2_size = None,
            GPUindex = GPUindex)
    elif classifier=="LSTM2":
        if kind=="from-scratch":
            output_size = len(classes)
        if kind=="incremental":
            output_size = n_classes
        if kind=="no-model-file":
            output_size = len(classes)
        net = classifier_LSTM(
            False,
            input_size = channel,
            lstm_layers = 1,
            lstm_size = 128,
            output1_size = output_size,
            output2_size = None,
            GPUindex = GPUindex)
    elif classifier=="LSTM3":
        if kind=="from-scratch":
            output_size = len(classes)
        if kind=="incremental":
            output_size = n_classes
        if kind=="no-model-file":
            output_size = len(classes)
        net = classifier_LSTM(
            True,
            input_size = channel,
            lstm_layers = 1,
            lstm_size = 128,
            output1_size = output_size,
            output2_size = None,
            GPUindex = GPUindex)
    elif classifier=="LSTM4":
        if kind=="from-scratch":
            output_size = len(classes)
        if kind=="incremental":
            output_size = n_classes
        if kind=="no-model-file":
            output_size = len(classes)
        net = classifier_LSTM(
            True,
            input_size = channel,
            lstm_layers = 1,
            lstm_size = 128,
            output1_size = 128,
            output2_size = output_size,
            GPUindex = GPUindex) 
    elif classifier=="Stacked_BiLSTM":
        if kind=="from-scratch":
            output_size = len(classes)
        if kind=="incremental":
            output_size = n_classes
        if kind=="no-model-file":
            output_size = len(classes)
        net = classifier_Stacked_BiLSTM(
            True,
            input_size = channel,
            lstm_layers =4,
            lstm_size = 128,
            output1_size = 128,
            output2_size = output_size,
            GPUindex = GPUindex)        
    elif classifier=="LSTM_per_channel":
        if kind=="from-scratch":
            output_size = len(classes)
        if kind=="incremental":
            output_size = n_classes
        if kind=="no-model-file":
            output_size = len(classes)
        net = classifier_LSTM_per_channel(
            channels_num = channel,
            lstm_layers = 1,
            lstm_full1_size = 128,
            lstm_full2_size= 64,
            output_size= output_size,
            GPUindex = GPUindex)
    elif classifier=="CNN":
        if length<min_CNN:
            return
        if kind=="from-scratch":
            output_size = len(classes)
        if kind=="incremental":
            output_size = n_classes
        if kind=="no-model-file":
            output_size = len(classes)
        net = classifier_CNN(
            in_channel = channel,
            timesteps = length,
                output_size = output_size)
    elif classifier=="EEGNet":
        if length<min_CNN:
            return
        if kind=="from-scratch":
            output_size = len(classes)
        if kind=="incremental":
            output_size = n_classes
        if kind=="no-model-file":
            output_size = len(classes)
        net = classifier_EEGNet(channel, length)
    elif classifier=="SyncNet":
        if length<min_CNN:
            return 
        if kind=="from-scratch":
            output_size = len(classes)
        if kind=="incremental":
            output_size = n_classes
        if kind=="no-model-file":
            output_size = len(classes)
        net = classifier_SyncNet(channel, length)
    elif classifier=="EEGChannelNet":
        if length<min_CNN:
            return
        if kind=="from-scratch":
            output_size = len(classes)
        if kind=="incremental":
            output_size = n_classes
        if kind=="no-model-file":
            output_size = len(classes)
        net = classifier_EEGChannelNet(channel, length)
    print("DONE: CREATE TORCH CLASSIFIER")
    print(net)
    nonclasses = [i for i in range(output_size) if i not in classes]
    return net, nonclasses