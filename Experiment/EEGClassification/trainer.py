import torch
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt 
import copy
import time
import os 

def fit(train_loader, val_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, device, log_interval, log_path_dir, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    start = time.time()
    train_losses = []  # List to store training losses for plotting
    val_losses = []  # List to store validation losses for plotting
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):

        # Train stage
        train_loss, train_acc = train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics)
        scheduler.step()

        message = 'Epoch: {}/{}. Train set: Average loss: {:.6f}. Accuracy: {:.4f}'.format(epoch + 1, n_epochs, train_loss, train_acc)
        # for metric in metrics:
        #     message += '\t{}: {}'.format(metric.name(), metric.value())
        # logger.info(message)

        val_loss, val_acc = test_epoch(val_loader, model, loss_fn, device, metrics)
        # val_loss /= len(val_loader)

        message += '\n\tValidation set: Average loss: {:.6f}. Accuracy: {:.4f}'.format(val_loss, val_acc)
        # for metric in metrics:
        #     message += '\t{}: {}'.format(metric.name(), metric.value())

        logger.info(message)
        train_losses.append(train_loss)  # Append training loss to list for plotting
        val_losses.append(val_loss)  # Append validation loss to list for plotting
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        if (val_acc > best_val_acc):
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 10 == 0 and (epoch+1) != n_epochs: # Epoch 10, 20, 30, 40, 50
            logger.info(f"Best val accuracy: {best_val_acc:.4f}")
            plot_losses(train_losses, val_losses, train_accs, val_accs, epoch + 1, log_path_dir)

    time_elapsed = time.time() - start
    logger.info('=====================================')
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info(f"Best Val Accuracy: {best_val_acc:.4f}")
    logger.info('***********')
    test_loss, test_acc = test_epoch(test_loader, model, loss_fn, device, metrics)
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info('=====================================')
    
    model.load_state_dict(best_model_weights)
    model_path = os.path.join(log_path_dir, f"model.pth")
    torch.save(model.state_dict(), model_path)    

    plot_losses(train_losses, val_losses, train_accs, val_accs, n_epochs, log_path_dir)  # Plot losses after the final 

    
def plot_losses(train_losses, val_losses, train_accs, val_accs, n_epochs, save_path_dir):
    save_fig_losses = os.path.join(save_path_dir, 'plot_losses.png')
    save_fig_accs = os.path.join(save_path_dir, 'plot_accuracy.png')
    plt.figure()
    plt.plot(range(1, n_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, n_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    # plt.xticks()
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(save_fig_losses)

    plt.figure()
    plt.plot(range(1, n_epochs + 1), train_accs, label='Train Accuracy')
    plt.plot(range(1, n_epochs + 1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(save_fig_accs)


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics):
    # for metric in metrics:
    #     metric.reset()

    model.train()
    losses = []
    
    running_loss = 0
    running_acc = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        # print(f"Device: {device}")
        # print(f"Batch {batch_idx}, batch_size: {len(targets)}")
        # print(f"EEG size: {data[0].size()}")
        # print(f"Image size: {data[1].size()}")
        # print(target)
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()

        # # WARNING: For inception_v3
        # if is_inception:
        #     # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
        #     outputs = model(data)
        #     loss1 = loss_fn(outputs, targets)
        #     # loss2 = loss_fn(aux_outputs, targets)
        #     loss = loss1 
        # else:
        #     outputs = model(data)
        #     loss = loss_fn(outputs, targets)
        outputs = model(data)
        loss = loss_fn(outputs, targets)

        _, preds = torch.max(outputs, 1)

        losses.append(loss.item())
        running_loss += loss.item()
        running_acc += (torch.sum(preds == targets) / len(targets)).double().item()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_acc / len(train_loader)
    return epoch_loss, epoch_acc


def test_epoch(data_loader, model, loss_fn, device, metrics):
    with torch.no_grad():
        # for metric in metrics:
        #     metric.reset()
        model.eval()
        running_loss = 0
        running_acc = 0
        for batch_idx, (data, targets) in enumerate(data_loader):
            # print(f"Batch {batch_idx}, batch_size: {len(target)}")
            # print(f"EEG size: {data[0].size()}")
            # print(f"Image size: {data[1].size()}")
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            loss = loss_fn(outputs, targets)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_acc += (torch.sum(preds == targets) / len(targets)).double().item()

            # for metric in metrics:
            #     metric(outputs, target, loss_outputs)
        epoch_loss = running_loss / len(data_loader)
        epoch_acc = running_acc / len(data_loader)

    return epoch_loss, epoch_acc