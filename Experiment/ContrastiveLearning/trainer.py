import torch
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt 
import os 

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, device, log_interval, log_path_dir, is_inception, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    train_losses = []  # List to store training losses for plotting
    val_losses = []  # List to store validation losses for plotting
    
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, is_inception, metrics)
        scheduler.step()

        message = 'Epoch: {}/{}. Train set: Average loss: {:.6f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        logger.info(message)

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, device, metrics)
        val_loss /= len(val_loader)

        message = 'Epoch: {}/{}. Validation set: Average loss: {:.6f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        logger.info(message)
        train_losses.append(train_loss)  # Append training loss to list for plotting
        val_losses.append(val_loss)  # Append validation loss to list for plotting
        if (epoch + 1) % 10 == 0: # Epoch 10, 20, 30, 40, 50
            plot_losses(train_losses, val_losses, epoch + 1, log_path_dir)
            model_path = os.path.join(log_path_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)

    # plot_losses(train_losses, val_losses, n_epochs, save_fig_train_val)  # Plot losses after the final 
    
def plot_losses(train_losses, val_losses, n_epochs, save_path_dir):
    save_fig_train_val = os.path.join(save_path_dir, 'train_val_losses.png')
    save_fig_train = os.path.join(save_path_dir, 'train_losses.png')
    plt.figure()
    plt.plot(range(1, n_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, n_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    # plt.xticks()
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(save_fig_train_val)

    plt.figure()
    plt.plot(range(1, n_epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(save_fig_train)


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, is_inception, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # print(f"Device: {device}")
        # print(f"Batch {batch_idx}, batch_size: {len(target)}")
        # print(f"EEG size: {data[0].size()}")
        # print(f"Image size: {data[1].size()}")
        # print(target)
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if device:
            data = tuple(d.to(device) for d in data)
            if target is not None:
                target = target.to(device)


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        # WARNING: For inception_v3, must include this if-statement. Otherwise skip it
        if (is_inception):
            if len(outputs) == 2: # (eeg, image)
                eeg, img = outputs
                img_logits = img.logits
                outputs = (eeg, img_logits)
            elif len(outputs) == 3: # (eeg, image_pos, image_neg)
                eeg, image_pos, image_neg = outputs
                # img_logits = img.logits
                outputs = (eeg, image_pos.logits, image_neg.logits)
        loss_inputs = outputs

        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            logger.info(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, device, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            # print(f"Batch {batch_idx}, batch_size: {len(target)}")
            # print(f"EEG size: {data[0].size()}")
            # print(f"Image size: {data[1].size()}")
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if device:
                data = tuple(d.to(device) for d in data)
                if target is not None:
                    target = target.to(device)

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            # if len(outputs) == 3: # (eeg, image_pos, image_neg)
            #     eeg, image_pos, image_neg = outputs
                
            
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            # print(f"Val loss output: {loss_outputs}")
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics