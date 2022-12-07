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
def net_trainer(
        net, loaders, opt, channel_idx, nonclasses, pretrain, train, save):
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
        for epoch in range(1, opt.epochs+1):
            print("epoch", epoch)
            # Initialize loss/accuracy variables
            losses = {"train": 0.0, "val": 0.0, "test": 0.0}
            corrects = {"train": 0.0, "val": 0.0, "test": 0.0}
            counts = {"train": 0.0, "val": 0.0, "test": 0.0}
            # Adjust learning rate for SGD
            if opt.optim=="SGD":
                lr = opt.learning_rate*(opt.learning_rate_decay_by**
                                        (epoch//opt.learning_rate_decay_every))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            # Process each split
            for split in ("train", "val", "test"):
                # Set network mode
                if split=="train":
                    net.train()
                else:
                    net.eval()
                # Process all split batches
                for i, (input, target) in enumerate(loaders[split]):
                    # Check CUDA
                    # async has become a reserved keyword in Python
                    # => async changes to non-blocking
                    if not opt.no_cuda:
                        if channel_idx is None:
                            input = input.cuda(opt.GPUindex, non_blocking = True)
                            target = target.cuda(opt.GPUindex, non_blocking = True)
                        else:
                            input = input[:, :, channel_idx].cuda(
                                opt.GPUindex, non_blocking = True)
                            target = target.cuda(opt.GPUindex, non_blocking = True)
                    # Wrap for autograd
                    if split == "train":
                        input = Variable(input)
                        target = Variable(target)
                        # Forward
                        output = net(input)
                        loss = F.cross_entropy(output, target)
                        losses[split] += loss.item()
                        # Compute accuracy
                        output.data[:, nonclasses] = minus_infinity
                        _, pred = output.data.max(1)
                        corrects[split] += pred.eq(target.data).sum().float()
                        counts[split] += input.data.size(0)
                        # Backward and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()


                        if ((i+1)%1000 ==0):
                            print(f"Batch {i+1}: Loss={loss.item()}; accuracy={corrects[split]/counts[split]}")

                    else:
                        with torch.no_grad():
                            # Forward
                            output = net(input)
                            loss = F.cross_entropy(output, target)
                            losses[split] += loss.item()
                            # Compute accuracy
                            output.data[:, nonclasses] = minus_infinity
                            _, pred = output.data.max(1)
                            corrects[split] += (
                                pred.eq(target.data).sum().float())
                            counts[split] += input.data.size(0)
        if save is not None:
            torch.save(net.state_dict(), save+".pth")
    else:
        # Initialize loss/accuracy variables
        losses = {"val": 0.0, "test": 0.0}
        corrects = {"val": 0.0, "test": 0.0}
        counts = {"val": 0.0, "test": 0.0}
        # Process each split
        for split in ("val", "test"):
            # Set network mode
            net.eval()
            # Process all split batches
            for i, (input, target) in enumerate(loaders[split]):
                # Check CUDA
                if not opt.no_cuda:
                    if channel_idx is None:
                        input = input.cuda(opt.GPUindex, non_blocking = True)
                        target = target.cuda(opt.GPUindex, non_blocking = True)
                    else:
                        input = input[:, :, channel_idx].cuda(
                            opt.GPUindex, non_blocking = True)
                        target = target.cuda(opt.GPUindex, non_blocking = True)
                with torch.no_grad():
                    # Forward
                    output = net(input)
                    loss = F.cross_entropy(output, target)
                    losses[split] += loss.item()
                    # Compute accuracy
                    output.data[:, nonclasses] = minus_infinity
                    _, pred = output.data.max(1)
                    corrects[split] += (
                        pred.eq(target.data).sum().float())
                    counts[split] += input.data.size(0)
    return ((corrects["val"]/counts["val"]).data.cpu().item(),
            (corrects["test"]/counts["test"]).data.cpu().item(),
            int(counts["val"]),
            int(counts["test"]))