import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from loguru import logger
import matplotlib.pyplot as plt
import os

def trainer_DCGAN(train_loader_stage1, train_loader_stage2, val_loader, netG, netD, criterion, optimizer_G, optimizer_D, is_pretrained_stage1, scheduler, log_path_dir, args):
    # Set the parameters
    latent_dim = args.latent_dim #100
    eeg_dim = args.eeg_dim # 128
    device = args.device

    # Set the training parameters
    num_epochs_stage1 = args.num_epochs_stage1 # 100
    num_epochs_stage2 = args.num_epochs_stage2 # 50
    log_interval = args.log_interval

    eval_noise = torch.normal(mean=0.0, std=1.0, size=(args.batch_size, latent_dim)).to(device)

    if not is_pretrained_stage1:
        # Training stage 1: Train non-conditional GAN on images without EEG data
        D_losses_stage1 = []
        G_losses_stage1 = []
        # Training loop
        for epoch in range(num_epochs_stage1):
            netG.train()
            netD.train()

            running_loss_G = 0.0
            running_loss_D = 0.0
            for i, (real_images, target) in enumerate(train_loader_stage1):
                real_images = real_images.to(device)
                target = target.to(device)

                # Extract Batch size
                N = real_images.size(0)
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ## For stage1, y_c, y_w is set to zeros
                ############################

                ## Train with all-real batch

                netD.zero_grad()
                ## For stage1, condition vector is set to zeros for netG
                real_labels = torch.ones(N, 1).to(device)
                ## For stage1, condition vector is set to zeros for netD
                real_outputs = netD(real_images, torch.zeros(N, eeg_dim).to(device))
                # CANNOT combine the losses to do backpropagation, must go separately!!!
                loss_D_real = criterion(real_outputs, real_labels) 
                loss_D_real.backward()
                D_x = real_outputs.mean().item()

                ## Train with all-fake batch
                noise = torch.normal(mean=0.0, std=1.0, size=(N, latent_dim)).to(device)
                fake_images = netG(noise, torch.zeros(N, eeg_dim).to(device))
                fake_labels = torch.zeros(N, 1).to(device)
                fake_outputs = netD(fake_images.detach(), torch.zeros(N, eeg_dim).to(device))
                loss_D_fake = criterion(fake_outputs, fake_labels)
                loss_D_fake.backward()
                D_G_z1 = fake_outputs.mean().item()

                loss_D = loss_D_real + loss_D_fake
                optimizer_D.step()
                
                ############################
                # (2) Update G network: maximize log(D(G(z))) => minimize negative log-likelihood
                ## For stage1, y_w is set to zeros
                ############################
                netG.zero_grad()
                fake_outputs = netD(fake_images, torch.zeros(N, eeg_dim).to(device))                
                loss_G = criterion(fake_outputs, real_labels)
                loss_G.backward()
                D_G_z2 = fake_outputs.mean().item()
                optimizer_G.step()

                running_loss_G += loss_G
                running_loss_D += loss_D
                
                # # Print training progress
                # if (i + 1) % log_interval == 0:
                #     logger.info(f"Stage 1 - Epoch [{epoch+1}/{num_epochs_stage1}], Step [{i+1}/{len(train_loader_stage1)}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
            epoch_loss_G = running_loss_G / len(train_loader_stage1)
            epoch_loss_D = running_loss_D / len(train_loader_stage1)
            D_losses_stage1.append(epoch_loss_D.item())
            G_losses_stage1.append(epoch_loss_G.item())
            logger.info(f"Stage 1 - Epoch [{epoch+1}/{num_epochs_stage1}], Loss D: {epoch_loss_D.item():.4f}, Loss G: {epoch_loss_G.item():.4f}, D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")
        torch.save(netG.state_dict(), f"{log_path_dir}/netG_stage1.pth")
        torch.save(netD.state_dict(), f"{log_path_dir}/netD_stage1.pth")
        plot_losses(D_losses_stage1, 100, log_path_dir, "Discriminator Loss Stage 1")
        plot_losses(G_losses_stage1, 100, log_path_dir, "Generator Loss Stage 1")

    # Training stage 2: Train GAN on images with EEG data
    D_losses_stage2 = []
    G_losses_stage2 = []
    # Training loop
    for epoch in range(num_epochs_stage2):
        train_loss_G, train_loss_D, D_xc_yc, D_xc_yw, D_G_z1, D_G_z2, train_real_images, train_fake_images = train_GAN_stage2(train_loader_stage2, netG, netD, criterion, optimizer_G, optimizer_D, args)
        D_losses_stage2.append(train_loss_D.item())
        G_losses_stage2.append(train_loss_G.item())
        logger.info(f"Stage 2 - Epoch [{epoch+1}/{num_epochs_stage2}], Loss D: {train_loss_D.item():.4f}, Loss G: {train_loss_G.item():.4f}, D(x): {D_xc_yc:.4f} | {D_xc_yw:.4f}, D(G(z)): {D_G_z1:.4f} | {D_G_z2:.4f}")
        # Checkpoint
        if (epoch+1) % log_interval == 0 and (epoch+1) != num_epochs_stage2:
            save_image(train_real_images, f"{log_path_dir}/real_images_epoch_{epoch+1}.png")
            save_image(train_fake_images, f"{log_path_dir}/fake_images_epoch_{epoch+1}.png")

            val_loss_G, val_loss_D, D_xc_yc, D_xc_yw, D_G_z1, D_G_z2, val_real_images, val_fake_images = test_GAN_stage2(eval_noise, val_loader, netG, netD, criterion, args)
            logger.info(f"Stage 2 - Validation, Loss D: {val_loss_D.item():.4f}, Loss G: {val_loss_G.item():.4f}, D(x): {D_xc_yc:.4f} | {D_xc_yw:.4f}, D(G(z)): {D_G_z1:.4f} | {D_G_z2:.4f}")
            save_image(val_real_images, f"{log_path_dir}/real_images_val.png")
            save_image(val_fake_images, f"{log_path_dir}/fake_images_val.png")

            torch.save(netG.state_dict(), f"{log_path_dir}/netG_stage2_epoch_{epoch+1}.pth")
            torch.save(netD.state_dict(), f"{log_path_dir}/netD_stage2_epoch_{epoch+1}.pth")
    torch.save(netG.state_dict(), f"{log_path_dir}/netG_stage2_epoch_{num_epochs_stage2}.pth")
    torch.save(netG.state_dict(), f"{log_path_dir}/netG_stage2_epoch_{num_epochs_stage2}.pth")
    plot_losses(D_losses_stage2, num_epochs_stage2, log_path_dir, "Discriminator Loss Stage 2")
    plot_losses(G_losses_stage2, num_epochs_stage2, log_path_dir, "Generator Loss Stage 2")
    

def train_GAN_stage2(data_loader, netG, netD, criterion, optimizer_G, optimizer_D, args):
    device = args.device
    latent_dim = args.latent_dim #100

    running_loss_G = 0.0
    running_loss_D = 0.0
    netG.train()
    netD.train()
    for i, (data, target) in enumerate(data_loader):
        real_images, avg_eeg_pos, avg_eeg_neg = data
        real_images = real_images.to(device)
        avg_eeg_pos, avg_eeg_neg = avg_eeg_pos.to(device), avg_eeg_neg.to(device)
        target = target.to(device)
        
        # Extract Batch size
        N = real_images.size(0)

        ############################
        # Train the netD: maximize log(D(x_c|y_c)) + log(1-D(x_c|y_w)) + log(1-D(x_w|y_w))
        ## For stage2, y_c, y_w is average eeg embeddings
        ############################

        ## In the 2nd stage of GAN training, we train the netD with 3 samples
        ### (1) Real images with correct condition (x_c, y_c)
        ### (2) Real images with wrong condition (x_c, y_w)
        ### (3) Fake images with wrong condition

        ## Train with real images with correct condition

        netD.zero_grad()
        real_labels = torch.ones(N, 1).to(device)
        fake_labels = torch.zeros(N, 1).to(device)

        out_xc_yc = netD(real_images, avg_eeg_pos)
        loss_D_xc_yc = criterion(out_xc_yc, real_labels) 
        loss_D_xc_yc.backward()
        D_xc_yc = out_xc_yc.mean().item()

        ## Train with real images with wrong condition batch

        out_xc_yw = netD(real_images, avg_eeg_neg)
        loss_D_xc_yw = criterion(out_xc_yw, fake_labels)
        loss_D_xc_yw.backward()
        D_xc_yw = out_xc_yw.mean().item()
        
        ## Train with all-fake batch

        noise = torch.normal(mean=0.0, std=1.0, size=(N, latent_dim)).to(device)
        fake_images = netG(noise, avg_eeg_pos)
        
        out_xw_yw = netD(fake_images.detach(), avg_eeg_neg)
        
        loss_D_xw_yw =  criterion(out_xw_yw, fake_labels)
        loss_D_xw_yw.backward()
        D_G_z1 = out_xw_yw.mean().item()
        loss_D = loss_D_xc_yc + loss_D_xc_yw + loss_D_xw_yw
        optimizer_D.step()
        
        ############################
        # Train the netG: maximize log(D(x_w|y_w)) 
        ## For stage2, y_w is average eeg embeddings
        ############################
        netG.zero_grad()
        fake_outputs = netD(fake_images, avg_eeg_neg)
        
        loss_G = criterion(fake_outputs, real_labels)
        loss_G.backward()
        D_G_z2 = fake_outputs.mean().item()
        optimizer_G.step()
        
        running_loss_G += loss_G
        running_loss_D += loss_D
        # # Print training progress
        # if (i + 1) % log_interval == 0:
        #     logger.info(f"Stage 2 - Epoch [{epoch+1}/{num_epochs_stage2}], Step [{i+1}/{len(train_loader_stage2)}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
    epoch_loss_G = running_loss_G / len(data_loader)
    epoch_loss_D = running_loss_D / len(data_loader)
    return epoch_loss_G, epoch_loss_D, D_xc_yc, D_xc_yw, D_G_z1, D_G_z2, real_images, fake_images

def test_GAN_stage2(eval_noise, data_loader, netG, netD, criterion, args):
    device = args.device
    latent_dim = args.latent_dim #100

    running_loss_G = 0.0
    running_loss_D = 0.0
    with torch.no_grad():
        netG.eval()
        netD.eval()
        for i, (data, target) in enumerate(data_loader):
            if i>0: #only evaluate images of the first batch
                break
            real_images, avg_eeg_pos, avg_eeg_neg = data
            real_images = real_images.to(device)
            avg_eeg_pos, avg_eeg_neg = avg_eeg_pos.to(device), avg_eeg_neg.to(device)
            target = target.to(device)
            
            # Extract Batch size
            N = real_images.size(0)

            ############################
            # Train the netD: maximize log(D(x_c|y_c)) + log(1-D(x_c|y_w)) + log(1-D(x_w|y_w))
            ## For stage2, y_c, y_w is average eeg embeddings
            ############################

            ## In the 2nd stage of GAN training, we train the netD with 3 samples
            ### (1) Real images with correct condition (x_c, y_c)
            ### (2) Real images with wrong condition (x_c, y_w)
            ### (3) Fake images with wrong condition

            ## Train with real images with correct condition

            real_labels = torch.ones(N, 1).to(device)
            fake_labels = torch.zeros(N, 1).to(device)
            out_xc_yc = netD(real_images, avg_eeg_pos)
            loss_D_xc_yc = criterion(out_xc_yc, real_labels) 
            D_xc_yc = out_xc_yc.mean().item()

            ## Train with real images with wrong condition batch

            out_xc_yw = netD(real_images, avg_eeg_neg)
            loss_D_xc_yw = criterion(out_xc_yw, fake_labels)
            D_xc_yw = out_xc_yw.mean().item()
            
            ## Train with all-fake batch

            # noise = torch.normal(mean=0.0, std=1.0, size=(N, latent_dim)).to(device)
            fake_images = netG(eval_noise, avg_eeg_pos)
            
            out_xw_yw = netD(fake_images.detach(), avg_eeg_neg)            
            loss_D_xw_yw =  criterion(out_xw_yw, fake_labels)
            D_G_z1 = out_xw_yw.mean().item()
            loss_D = loss_D_xc_yc + loss_D_xc_yw + loss_D_xw_yw
            
            ############################
            # Train the netG: maximize log(D(x_w|y_w)) 
            ## For stage2, y_w is average eeg embeddings
            ############################
            fake_outputs = netD(fake_images, avg_eeg_neg)
            
            loss_G = criterion(fake_outputs, real_labels)
            D_G_z2 = fake_outputs.mean().item()
            
            running_loss_G += loss_G
            running_loss_D += loss_D
        epoch_loss_G = running_loss_G / len(data_loader)
        epoch_loss_D = running_loss_D / len(data_loader)
        return epoch_loss_G, epoch_loss_D, D_xc_yc, D_xc_yw, D_G_z1, D_G_z2, real_images, fake_images
            
def plot_losses(losses, n_epochs, save_path_dir, info):
    save_fig_losses = os.path.join(save_path_dir, f"{info}.png")
    plt.figure()
    plt.plot(range(1, n_epochs + 1), losses, label=info)
    plt.xlabel('Epoch')
    # plt.xticks()
    plt.ylabel('Loss')
    plt.title(info)
    plt.legend()
    plt.savefig(save_fig_losses)