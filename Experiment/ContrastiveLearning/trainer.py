import torch
import numpy as np
from loguru import logger

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, device, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics)
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


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics):
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
            
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics