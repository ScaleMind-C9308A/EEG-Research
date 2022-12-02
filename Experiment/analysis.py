from torch.utils.data import DataLoader
from data_loader import EEGDataset, Splitter, SplitterWithData
from EEG_Encoder.LSTM import classifier_LSTM
from EEG_Encoder.CNN import classifier_CNN
from EEG_Encoder.EEGNet import classifier_EEGNet
from EEG_Encoder.SyncNet import classifier_SyncNet
from EEG_Encoder.EEGChannelNet import classifier_EEGChannelNet
from EEG_Encoder.net_trainer import net_trainer
from p_values import *
def analysis(iv,
             offset,
             fold,
             eeg_dataset,
             splits_path,
             total,
             classes,
             classifier,
             batch_size,
             GPUindex,
             length, # 512
             channel, # 96
             min_CNN,
             opt,
             kind):
    val, test, samples = 0.0, 0.0, 0
    for split_num in range(fold):
        model_path = (iv+
                      "-"+
                      classifier+
                      "-"+
                      str(length)+
                      "-"+
                      str(channel)+
                      "-"+
                      str(split_num))
        # Load dataset
        dataset = EEGDataset(iv, eeg_dataset, classifier, map_idx = None)
        print("DONE: LOAD DATASET")
        # Create loaders for LSTM/MLP/CNN/SCNN/EEGNet/SyncNet/EEGChannelNet
        if kind=="from-scratch":
            relabel = True
        if kind=="incremental":
            relabel = False
        if kind=="no-model-file":
            relabel = True
        loaders = {split: DataLoader(
            SplitterWithData(iv,
                        dataset,
                        splits_path,
                        classes,
                        split_num,
                        split,
                        relabel),
            batch_size = batch_size,
            drop_last = False,
            shuffle = True)
                for split in ["train", "val", "test"]}
        channel_idx = None    
        print("DONE: Create loaders for model")            
        # Training
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
                output_size = total
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
                output_size = total
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
                output_size = total
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
        elif classifier=="CNN":
            if length<min_CNN:
                break
            if kind=="from-scratch":
                output_size = len(classes)
            if kind=="incremental":
                output_size = total
            if kind=="no-model-file":
                output_size = len(classes)
            net = classifier_CNN(
                in_channel = channel,
                num_points = length,
                    output_size = output_size)
        elif classifier=="EEGNet":
            if length<min_CNN:
                break
            if kind=="from-scratch":
                output_size = len(classes)
            if kind=="incremental":
                output_size = total
            if kind=="no-model-file":
                output_size = len(classes)
            net = classifier_EEGNet(channel, length)
        elif classifier=="SyncNet":
            if length<min_CNN:
                break
            if kind=="from-scratch":
                output_size = len(classes)
            if kind=="incremental":
                output_size = total
            if kind=="no-model-file":
                output_size = len(classes)
            net = classifier_SyncNet(channel, length)
        elif classifier=="EEGChannelNet":
            if length<min_CNN:
                break
            if kind=="from-scratch":
                output_size = len(classes)
            if kind=="incremental":
                output_size = total
            if kind=="no-model-file":
                output_size = len(classes)
            net = classifier_EEGChannelNet(channel, length)
        print("DONE: CREATE TORCH CLASSIFIER")
        print(net)
        nonclasses = [i for i in range(output_size) if i not in classes]
        if kind=="from-scratch":
            accuracy_val, accuracy_test, counts_val, counts_test = net_trainer(
                    net,
                    loaders,
                    opt,
                    channel_idx,
                    nonclasses,
                    None,
                    True,
                    model_path)
        if kind=="incremental":
            accuracy_val, accuracy_test, counts_val, counts_test = net_trainer(
                    net,
                    loaders,
                    opt,
                    channel_idx,
                    nonclasses,
                    model_path,
                    True,
                    None)
        if kind=="no-model-file":
            accuracy_val, accuracy_test, counts_val, counts_test = net_trainer(
                    net,
                    loaders,
                    opt,
                    channel_idx,
                    nonclasses,
                    None,
                    True,
                    None)
        val += accuracy_val
        test += accuracy_test
        samples += counts_val+counts_test
    accuracy = (val+test)/(fold*2)
    return accuracy, p_value(accuracy, samples, len(classes))