import torch
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt 
import os 
import torchvision.utils as vutils

def fit(train_loader, val_loader, netG, netD, weights_init, loss_fn, optimizer, scheduler, n_epochs, device, log_interval, log_path_dir,lr, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    REAL_LABEL = 1
    FAKE_LABEL = 0
    Z_DIM = 100
    BATCH_SIZE = 128
    netG = netG.to(device)
    netG.apply(weights_init)
    print(netG)

    netD = netD.to(device)
    netD.apply(weights_init)
    print(netD)


    viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)

    optimizerD = optimizer(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optimizer(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        for i, data in enumerate(train_loader):
            x_real = data[0].to(device)
            real_label = torch.full((x_real.size(0),), REAL_LABEL, device=device)
            fake_label = torch.full((x_real.size(0),), FAKE_LABEL, device=device)

            # Update D with real data
            netD.zero_grad()
            y_real = netD(x_real)
            loss_D_real = loss_fn(y_real, real_label)
            loss_D_real.backward()

            # Update D with fake data
            z_noise = torch.randn(x_real.size(0), Z_DIM, 1, 1, device=device)
            x_fake = netG(z_noise)
            y_fake = netD(x_fake.detach())
            loss_D_fake = loss_fn(y_fake, fake_label)
            loss_D_fake.backward()
            optimizerD.step()

            # Update G with fake data
            netG.zero_grad()
            y_fake_r = netD(x_fake)
            loss_G = loss_fn(y_fake_r, real_label)
            loss_G.backward()
            optimizerG.step()

            if i % 100 == 0:
                print('Epoch {} [{}/{}] loss_D_real: {:.4f} loss_D_fake: {:.4f} loss_G: {:.4f}'.format(
                    epoch, i, len(train_loader),
                    loss_D_real.mean().item(),
                    loss_D_fake.mean().item(),
                    loss_G.mean().item()
                ))
                vutils.save_image(x_real, os.path.join(log_path_dir, 'real_samples.png'), normalize=True)
                with torch.no_grad():
                    viz_sample = netG(viz_noise)
                    vutils.save_image(viz_sample, os.path.join(log_path_dir, 'fake_samples_{}.png'.format(epoch)), normalize=True)
        torch.save(netG.state_dict(), os.path.join(log_path_dir, 'netG_{}.pth'.format(epoch)))
        torch.save(netD.state_dict(), os.path.join(log_path_dir, 'netD_{}.pth'.format(epoch)))