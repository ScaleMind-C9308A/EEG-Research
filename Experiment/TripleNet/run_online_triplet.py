from datasets import BalancedBatchSampler
# Set up the network and training parameters
# from networks import EmbeddingNet
import torch
from losses import OnlineTripletLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from metrics import AverageNonzeroTripletsMetric
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from trainer import net_trainer

def run():   
    """
    Detailed steps in run:
    Step 0: Setup and preprocessing
        - Define options (can use ArgParser)
            + eeg_dir_path, img_dir_path
            + lr, 
    Step 1: Set Dataloaders (data_loader.py)
        - From load_data(eeg_path, image_path, ...): 
            return train_dataloader, val_dataloader and test_dataloader
    Step 2: Set model (model.py)
    Step 3: Set loss_fn (losses.py)
    Step 4: Set optimizer (Adam/SGD)
    Step 5: Put all to net_trainer()
    """
    # We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
    

    margin = 1.
    embedding_net = EmbeddingNet()
    model = embedding_net
    if cuda:
        model.cuda()
    loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 150

    net_trainer(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AverageNonzeroTripletsMetric()])
