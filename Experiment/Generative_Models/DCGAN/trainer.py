import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger

def GAN_fit(train_loader_stage1, train_loader_stage2, val_loader, generator, discriminator, criterion, optimizer_G, optimizer_D, scheduler, log_path_dir, args):
    # Set the parameters
    latent_dim = args.latent_dim #100
    eeg_dim = args.eeg_dim # 128
    image_size = args.img_size # 64
    device = args.device

    # Set the training parameters
    num_epochs_stage1 = args.num_epochs_stage1 # 100
    num_epochs_stage2 = args.num_epochs_stage2 # 50
    log_interval = args.log_interval
    save_interval = args.save_interval

    # Training stage 1: Train non-conditional GAN on images without EEG data
    # (Assuming you have a dataset of images without EEG data named "dataset_stage1")

    # Training loop
    for epoch in range(num_epochs_stage1):
        for i, (real_images, target) in enumerate(train_loader_stage1):
            real_images = real_images.to(device)
            target = target.to(device)

            # Extract Batch size
            N = real_images.size(0)
            ############################
            # Train the discriminator: maximize log(D(x_c|y_c)) + log(1-D(x_c|y_w)) + log(1-D(x_w|y_w))
            ## For stage1, y_c, y_w is set to zeros
            ############################
            optimizer_D.zero_grad()
            ## For stage1, condition vector is set to zeros for generator
            noise = torch.normal(mean=0.0, std=1.0, size=(N, latent_dim)).to(device)
            fake_images = generator(noise, torch.zeros(N, eeg_dim).to(device))
            real_labels = torch.ones(N, 1).to(device)
            fake_labels = torch.zeros(N, 1).to(device)
            ## For stage1, condition vector is set to zeros for discriminator
            real_outputs = discriminator(real_images, torch.zeros(N, eeg_dim).to(device))
            fake_outputs = discriminator(fake_images.detach(), torch.zeros(N, eeg_dim).to(device))
            
            loss_D = criterion(real_outputs, real_labels) + criterion(fake_outputs, fake_labels)
            loss_D.backward()
            optimizer_D.step()
            
            ############################
            # Train the generator: minimize log(D(x_w|y_w)) 
            ## For stage1, y_w is set to zeros
            ############################
            optimizer_G.zero_grad()
            fake_outputs = discriminator(fake_images, torch.zeros(N, eeg_dim).to(device))
            
            loss_G = criterion(fake_outputs, real_labels)
            loss_G.backward()
            optimizer_G.step()
            
            # Print training progress
            if (i + 1) % log_interval == 0:
                logger.info(f"Stage 1 - Epoch [{epoch+1}/{num_epochs_stage1}], Step [{i+1}/{len(train_loader_stage1)}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        logger.info(f"Stage 1 - Epoch [{epoch+1}/{num_epochs_stage1}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        

    # Training stage 2: Train GAN on images with EEG data
    # (Assuming you have a dataset of images with EEG data named "dataset_stage2")

    # Training loop
    for epoch in range(num_epochs_stage2):
        for i, (data, target) in enumerate(train_loader_stage2):
            real_images, avg_eeg_pos, avg_eeg_neg = data
            real_images = real_images.to(device)
            avg_eeg_pos, avg_eeg_neg = avg_eeg_pos.to(device), avg_eeg_neg.to(device)
            target = target.to(device)
            
            # Extract Batch size
            N = real_images.size(0)

            ############################
            # Train the discriminator: maximize log(D(x_c|y_c)) + log(1-D(x_c|y_w)) + log(1-D(x_w|y_w))
            ## For stage2, y_c, y_w is average eeg embeddings
            ############################
            optimizer_D.zero_grad()
            noise = torch.normal(mean=0.0, std=1.0, size=(N, latent_dim)).to(device)
            fake_images = generator(noise, avg_eeg_pos)
            real_labels = torch.ones(N, 1).to(device)
            fake_labels = torch.zeros(N, 1).to(device)
            
            ## In the 2nd stage of GAN training, we train the discriminator with 3 samples
            ### (1) Real images with correct condition (x_c, y_c)
            ### (2) Real images with wrong condition (x_c, y_w)
            ### (3) Fake images with wrong condition
            real_w_correct_condition = discriminator(real_images, avg_eeg_pos)
            real_w_wrong_condition = discriminator(real_images, avg_eeg_neg)
            fake_w_wrong_condition = discriminator(fake_images.detach(), avg_eeg_neg)
            
            loss_D = criterion(real_w_correct_condition, real_labels) + criterion(real_w_wrong_condition, fake_labels) + criterion(fake_w_wrong_condition, fake_labels)
            loss_D.backward()
            optimizer_D.step()
            
            ############################
            # Train the generator: minimize log(D(x_w|y_w)) 
            ## For stage2, y_w is average eeg embeddings
            ############################
            optimizer_G.zero_grad()
            fake_outputs = discriminator(fake_images, avg_eeg_neg)
            
            loss_G = criterion(fake_outputs, real_labels)
            loss_G.backward()
            optimizer_G.step()
            
            # Print training progress
            if (i + 1) % log_interval == 0:
                logger.info(f"Stage 2 - Epoch [{epoch+1}/{num_epochs_stage2}], Step [{i+1}/{len(train_loader_stage2)}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        logger.info(f"Stage 2 - Epoch [{epoch+1}/{num_epochs_stage2}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        # Checkpoint
        if (epoch+1) % save_interval == 0:
            torch.save(generator.state_dict(), f"{log_path_dir}/netG_stage2_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"{log_path_dir}/netD_stage2_epoch_{epoch+1}.pth")