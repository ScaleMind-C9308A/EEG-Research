import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
from scipy.stats import mode

minus_infinity = float("-Inf")
##############################################################
# Network trainer
##############################################################
def train_step(model, dataloader, loss_fn, optimizer, device, print_every):
    # Put model in train mode
    model.train()
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        # 1. Forward pass
        output = model(X)
        # 2. Calculate  and accumulate loss
        loss = loss_fn(output, y)
        current_train_loss = loss.item()
        train_loss += loss.item() 
        # 3. Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Calculate and accumulate accuracy metric across all batches
        _, pred = output.data.max(1)
        current_corrects = pred.eq(y.data).sum().float()
        current_counts = X.data.size(0)
        current_train_acc = current_corrects / current_counts
        train_acc += current_train_acc
        # Print
        if ((batch+1)%print_every ==0):
            print(f"Train Batch {batch+1} (every {print_every} batch): Loss={current_train_loss:.4f}; accuracy={(train_acc/(batch+1)):.4f}")

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device, print_every):
    # Put model in eval mode
    model.eval() 
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            test_pred_logits = model(X)
            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            current_loss = loss.item()
            test_loss += current_loss
            # Calculate and accumulate accuracy
            _, pred = test_pred_logits.data.max(1)
            current_corrects = pred.eq(y.data).sum().float()
            current_counts = X.data.size(0)
            current_acc = current_corrects / current_counts
            test_acc += current_acc
#             # Print
#             if ((batch+1) % print_every == 0):                            
#                 print(f"Validation Batch {batch} (every {print_every} batch): Loss={current_loss:.4f}; accuracy= {(test_acc/(batch+1)):.4f}")
                
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def net_trainer(
        net, loaders, opt, channel_idx, nonclasses, pretrain, train, save, print_every_train=100, print_every_val=125):
    optimizer = getattr(torch.optim,
                        opt.optim)(net.parameters(),
                                   lr = opt.learning_rate)
    if pretrain is not None:
        net.load_state_dict(torch.load(pretrain+".pth", map_location = "cpu"))
    # Setup CUDA
    if not opt.no_cuda:
        net.cuda(opt.GPUindex)
    
    # Start training
    if train:
        # Create empty results dictionary
        results = {"train_loss": [], "train_acc": [],"val_loss": [],"val_acc": []}
        for epoch in range(1, opt.epochs+1):
            print("Epoch", epoch)         
            # Adjust learning rate for SGD
            if opt.optim=="SGD":
                lr = opt.learning_rate*(opt.learning_rate_decay_by**
                                        (epoch//opt.learning_rate_decay_every))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            # Process each split
            train_loss, train_acc = train_step(model=net,
                                      dataloader=loaders['train'],
                                      loss_fn=nn.CrossEntropyLoss(),
                                      optimizer=optimizer,
                                      device='cuda',
                                      print_every=print_every_train)

            val_loss, val_acc = test_step(model=net,
                                          dataloader=loaders['val'],
                                          loss_fn=nn.CrossEntropyLoss(),
                                          device='cuda',
                                          print_every=print_every_val)
                      
            # Summarize the training process
            print(f"Epoch {epoch} summary: train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")
            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)
        if save is not None:
            torch.save(net, save+".pth")
        
    else:
        results = {"val_acc": None, "test_acc": None}
        val_loss, val_acc = test_step(model=net,
                                          dataloader=loaders['val'],
                                          loss_fn=nn.CrossEntropyLoss(),
                                          optimizer=optimizer,
                                          device='cuda')
        test_loss, test_acc = test_step(model=net,
                                          dataloader=loaders['test'],
                                          loss_fn=nn.CrossEntropyLoss(),
                                          optimizer=optimizer,
                                          device='cuda')
        results["val_acc"] = val_acc
        results["test_acc"] = test_acc
            
    return results