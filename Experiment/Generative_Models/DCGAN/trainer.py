import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def GAN_fit(train_loader_stage1, train_loader_stage2, val_loader, generator, discriminator, criterion, optimizer_G, optimizer_D, scheduler, num_epochs_stage1, num_epochs_stage2, device, log_interval, log_path_dir):
    # Set the parameters
    latent_dim = 100
    eeg_dim = 128
    image_size = 64

    # Set the training parameters
    num_epochs_stage1 = 100
    num_epochs_stage2 = 50
    batch_size = 64
    learning_rate = 0.0002

    # Training stage 1: Train non-conditional GAN on images without EEG data
    # (Assuming you have a dataset of images without EEG data named "dataset_stage1")

    # Training loop
    for epoch in range(num_epochs_stage1):
        for i, real_images in enumerate(train_loader_stage1):
            real_images = real_images.to(device)
            
            # Train the discriminator
            optimizer_D.zero_grad()
            fake_images = generator(torch.randn(real_images.size(0), latent_dim).to(device), torch.zeros(real_images.size(0), eeg_dim).to(device))
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)
            
            real_outputs = discriminator(real_images, torch.zeros(real_images.size(0), eeg_dim).to(device))
            fake_outputs = discriminator(fake_images.detach(), torch.zeros(real_images.size(0), eeg_dim).to(device))
            
            loss_D = criterion(real_outputs, real_labels) + criterion(fake_outputs, fake_labels)
            loss_D.backward()
            optimizer_D.step()
            
            # Train the generator
            optimizer_G.zero_grad()
            fake_images = generator(torch.randn(real_images.size(0), latent_dim).to(device), torch.zeros(real_images.size(0), eeg_dim).to(device))
            fake_outputs = discriminator(fake_images, torch.zeros(real_images.size(0), eeg_dim).to(device))
            
            loss_G = criterion(fake_outputs, real_labels)
            loss_G.backward()
            optimizer_G.step()
            
            # Print training progress
            if (i + 1) % 10 == 0:
                print(f"Stage 1 - Epoch [{epoch+1}/{num_epochs_stage1}], Step [{i+1}/{len(train_loader_stage1)}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

    # Training stage 2: Train GAN on images with EEG data
    # (Assuming you have a dataset of images with EEG data named "dataset_stage2")

    # Training loop
    for epoch in range(num_epochs_stage2):
        for i, (real_images, eeg_features) in enumerate(train_loader_stage2):
            real_images = real_images.to(device)
            eeg_features = eeg_features.to(device)
            
            # Train the discriminator
            optimizer_D.zero_grad()
            fake_images = generator(torch.randn(real_images.size(0), latent_dim).to(device), eeg_features)
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)
            
            real_outputs = discriminator(real_images, eeg_features)
            fake_outputs = discriminator(fake_images.detach(), eeg_features)
            
            loss_D = criterion(real_outputs, real_labels) + criterion(fake_outputs, fake_labels)
            loss_D.backward()
            optimizer_D.step()
            
            # Train the generator
            optimizer_G.zero_grad()
            fake_images = generator(torch.randn(real_images.size(0), latent_dim).to(device), eeg_features)
            fake_outputs = discriminator(fake_images, eeg_features)
            
            loss_G = criterion(fake_outputs, real_labels)
            loss_G.backward()
            optimizer_G.step()
            
            # Print training progress
            if (i + 1) % 10 == 0:
                print(f"Stage 2 - Epoch [{epoch+1}/{num_epochs_stage2}], Step [{i+1}/{len(train_loader_stage2)}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
