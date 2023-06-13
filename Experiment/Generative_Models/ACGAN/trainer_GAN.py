import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from loguru import logger
import matplotlib.pyplot as plt
import os

from utils import compute_acc

def trainer_GAN(train_loader_stage1, train_loader_stage2, val_loader, netG, netD, dis_criterion, aux_criterion, optimizer_G, optimizer_D, is_pretrained_stage1, scheduler, log_path_dir, args):
    # Set the parameters
    latent_dim = args.latent_dim #100
    eeg_dim = args.eeg_dim # 128
    num_classes = args.num_classes # 10
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
            # netG.train()
            # netD.train()

            running_loss_G = 0.0
            running_loss_D = 0.0
            for i, (real_images, target) in enumerate(train_loader_stage1):
                real_images = real_images.to(device)
                target = target.to(device)
                
                # Extract Batch size
                N = real_images.size(0)
                

                ############################
                # Train the netD: maximize log(D(x_c|y_c)) + log(1-D(x_c|y_w)) + log(1-D(x_w|y_w))
                ############################

                ## In the 2nd stage of GAN training, we train the netD with 2 samples
                ### (1) Real images
                ### (2) Fake images 

                ## Train with real images with correct condition
                netD.train()
                # netG.eval()
                netD.zero_grad()
                real_labels = torch.ones(N, 1).to(device)
                fake_labels = torch.zeros(N, 1).to(device)

                dis_output, aux_output = netD(real_images)
                dis_errD_real = dis_criterion(dis_output, real_labels)
                aux_errD_real = aux_criterion(aux_output, target)
                
                errD_real = dis_errD_real + aux_errD_real
                # errD_real.backward()
                D_x = dis_output.mean().item()
                # compute the current classification accuracy
                aux_accuracy = compute_acc(aux_output, target)
                
                # Train with all-fake batch

                noise = torch.normal(mean=0.0, std=1.0, size=(N, latent_dim)).to(device)
                condition = F.one_hot(target, num_classes).float().to(device) 
                fake_images = netG(noise, condition)
                
                dis_output, aux_output = netD(fake_images.detach())
                dis_errD_fake = dis_criterion(dis_output, fake_labels)
                aux_errD_fake = aux_criterion(aux_output, target)
                errD_fake = dis_errD_fake + aux_errD_fake
                # errD_fake.backward()
                D_G_z1 = dis_output.mean().item()
                errD = errD_real + errD_fake
                errD.backward()
                optimizer_D.step()
                
                ############################
                # Train the netG: maximize log(D(x_w|y_w)) 
                ## For stage2, y_w is average eeg embeddings
                ############################
                netG.zero_grad()
                netD.eval()
                netG.train()
                noise = torch.normal(mean=0.0, std=1.0, size=(N, latent_dim)).to(device)
                condition = F.one_hot(target, num_classes).float().to(device)
                fake_images = netG(noise, condition)
                dis_output, aux_output = netD(fake_images)
                dis_errG = dis_criterion(dis_output, real_labels)
                aux_errG = aux_criterion(aux_output, target)
                errG = dis_errG + aux_errG
                # netD.zero_grad()
                # netG.zero_grad()
                errG.backward()
                D_G_z2 = dis_output.mean().item()
                optimizer_G.step()
                
                running_loss_G += errG
                running_loss_D += errD
                
            if (epoch+1) % log_interval == 0:
                save_image(real_images, f"{log_path_dir}/train_real_images_epoch_{epoch+1}.png")
                save_image(fake_images, f"{log_path_dir}/train_fake_images_epoch_{epoch+1}.png")
            epoch_loss_G = running_loss_G / len(train_loader_stage1)
            epoch_loss_D = running_loss_D / len(train_loader_stage1)
            D_losses_stage1.append(epoch_loss_D.item())
            G_losses_stage1.append(epoch_loss_G.item())
            logger.info(f"Stage 1 - Epoch [{epoch+1}/{num_epochs_stage1}], Loss D: {epoch_loss_D.item():.4f}, Loss G: {epoch_loss_G.item():.4f}, D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f} | {D_G_z2:.4f}, aux_acc: {aux_accuracy:.4f}")
        torch.save(netG.state_dict(), f"{log_path_dir}/netG_stage1.pth")
        torch.save(netD.state_dict(), f"{log_path_dir}/netD_stage1.pth")
        plot_losses(D_losses_stage1, num_epochs_stage1, log_path_dir, "Discriminator Loss Stage 1")
        plot_losses(G_losses_stage1, num_epochs_stage1, log_path_dir, "Generator Loss Stage 1")

    # # Training stage 2: Train GAN on images with EEG data
    # D_losses_stage2 = []
    # G_losses_stage2 = []
    # # Training loop
    # for epoch in range(num_epochs_stage2):
    #     train_loss_G, train_loss_D, D_x, D_G_z1, D_G_z2, aux_accuracy, train_real_images, train_fake_images = train_GAN_stage2(train_loader_stage2, netG, netD, dis_criterion, aux_criterion, optimizer_G, optimizer_D, args)
    #     D_losses_stage2.append(train_loss_D.item())
    #     G_losses_stage2.append(train_loss_G.item())
    #     logger.info(f"Stage 2 - Epoch [{epoch+1}/{num_epochs_stage2}], Loss D: {train_loss_D.item():.4f}, Loss G: {train_loss_G.item():.4f}, D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f} | {D_G_z2:.4f}, aux_acc: {aux_accuracy:.4f}")
    #     # Checkpoint
    #     if (epoch+1) % log_interval == 0:
    #         save_image(train_real_images, f"{log_path_dir}/train_real_images_epoch_{epoch+1}.png")
    #         save_image(train_fake_images, f"{log_path_dir}/train_fake_images_epoch_{epoch+1}.png")

    #         # val_loss_G, val_loss_D, D_x, D_G_z1, D_G_z2, aux_accuracy, val_real_images, val_fake_images = test_GAN_stage2(eval_noise, val_loader, netG, netD, dis_criterion, aux_criterion, args)
    #         # logger.info(f"Stage 2 - Validation, Loss D: {val_loss_D.item():.4f}, Loss G: {val_loss_G.item():.4f}, D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f} | {D_G_z2:.4f}, aux_acc: {aux_accuracy:.4f}")
    #         # save_image(val_real_images, f"{log_path_dir}/val_real_images.png")
    #         # save_image(val_fake_images, f"{log_path_dir}/val_fake_images.png")

    #         torch.save(netG.state_dict(), f"{log_path_dir}/netG_epoch_{epoch+1}.pth")
    #         torch.save(netD.state_dict(), f"{log_path_dir}/netD_epoch_{epoch+1}.pth")
    # plot_losses(D_losses_stage2, num_epochs_stage2, log_path_dir, "Discriminator Loss Stage 2")
    # plot_losses(G_losses_stage2, num_epochs_stage2, log_path_dir, "Generator Loss Stage 2")
    

def train_GAN_stage2(data_loader, netG, netD, dis_criterion, aux_criterion, optimizer_G, optimizer_D, args):
    device = args.device
    latent_dim = args.latent_dim #100

    running_loss_G = 0.0
    running_loss_D = 0.0
    netG.train()
    netD.train()
    for i, (data, target) in enumerate(data_loader):
        real_images, avg_eeg_pos = data
        real_images = real_images.to(device)
        avg_eeg_pos = avg_eeg_pos.to(device)
        target = target.to(device)
        
        # Extract Batch size
        N = real_images.size(0)

        ############################
        # Train the netD: maximize log(D(x_c|y_c)) + log(1-D(x_c|y_w)) + log(1-D(x_w|y_w))
        ############################

        ## In the 2nd stage of GAN training, we train the netD with 2 samples
        ### (1) Real images
        ### (2) Fake images 

        ## Train with real images with correct condition

        netD.zero_grad()
        real_labels = torch.ones(N, 1).to(device)
        fake_labels = torch.zeros(N, 1).to(device)

        dis_output, aux_output = netD(real_images)
        dis_errD_real = dis_criterion(dis_output, real_labels)
        aux_errD_real = aux_criterion(aux_output, target)
        
        errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()
        D_x = dis_output.mean().item()
        # compute the current classification accuracy
        aux_accuracy = compute_acc(aux_output, target)
        
        ## Train with all-fake batch

        noise = torch.normal(mean=0.0, std=1.0, size=(N, latent_dim)).to(device)
        fake_images = netG(noise, avg_eeg_pos)
        
        dis_output, aux_output = netD(fake_images.detach())
        dis_errD_fake = dis_criterion(dis_output, fake_labels)
        aux_errD_fake = aux_criterion(aux_output, target)
        errD_fake = dis_errD_fake + aux_errD_fake
        errD_fake.backward()
        D_G_z1 = dis_output.mean().item()
        errD = errD_real + errD_fake
        optimizer_D.step()
        
        ############################
        # Train the netG: maximize log(D(x_w|y_w)) 
        ## For stage2, y_w is average eeg embeddings
        ############################
        netG.zero_grad()

        dis_output, aux_output = netD(fake_images)
        dis_errG = dis_criterion(dis_output, real_labels)
        aux_errG = aux_criterion(aux_output, target)
        errG = dis_errG + aux_errG
        errG.backward()
        D_G_z2 = dis_output.mean().item()
        optimizer_G.step()
        
        running_loss_G += errG
        running_loss_D += errD
        # # Print training progress
        # if (i + 1) % log_interval == 0:
        #     logger.info(f"Stage 2 - Epoch [{epoch+1}/{num_epochs_stage2}], Step [{i+1}/{len(train_loader_stage2)}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
    epoch_loss_G = running_loss_G / len(data_loader)
    epoch_loss_D = running_loss_D / len(data_loader)
    return epoch_loss_G, epoch_loss_D, D_x, D_G_z1, D_G_z2, aux_accuracy, real_images, fake_images

def test_GAN_stage2(eval_noise, data_loader, netG, netD, dis_criterion, aux_criterion, args):
    device = args.device
    latent_dim = args.latent_dim #100

    running_loss_G = 0.0
    running_loss_D = 0.0
    with torch.no_grad():
        netG.eval()
        netD.eval()
        for i, (data, target) in enumerate(data_loader):
            if i>0: # only evaluate images of the first batch
                break
            real_images, avg_eeg_pos = data
            real_images = real_images.to(device)
            avg_eeg_pos = avg_eeg_pos.to(device)
            target = target.to(device)
            
            # Extract Batch size
            N = real_images.size(0)

            ############################
            # Train the netD: maximize log(D(x_c|y_c)) + log(1-D(x_c|y_w)) + log(1-D(x_w|y_w))
            ############################

            ## In the 2nd stage of GAN training, we train the netD with 2 samples
            ### (1) Real images
            ### (2) Fake images 

            ## Train with real images with correct condition

            real_labels = torch.ones(N, 1).to(device)
            fake_labels = torch.zeros(N, 1).to(device)

            dis_output, aux_output = netD(real_images)
            dis_errD_real = dis_criterion(dis_output, real_labels)
            aux_errD_real = aux_criterion(aux_output, target)
            
            errD_real = dis_errD_real + aux_errD_real
            D_x = dis_output.mean().item()
            # compute the current classification accuracy
            aux_accuracy = compute_acc(aux_output, target)
            
            ## Train with all-fake batch

            # noise = torch.normal(mean=0.0, std=1.0, size=(N, latent_dim)).to(device)
            fake_images = netG(eval_noise, avg_eeg_pos)
            
            dis_output, aux_output = netD(fake_images.detach())
            dis_errD_fake = dis_criterion(dis_output, fake_labels)
            aux_errD_fake = aux_criterion(aux_output, target)
            errD_fake = dis_errD_fake + aux_errD_fake
            D_G_z1 = dis_output.mean().item()
            errD = errD_real + errD_fake
            
            ############################
            # Train the netG: maximize log(D(x_w|y_w)) 
            ## For stage2, y_w is average eeg embeddings
            ############################

            dis_output, aux_output = netD(fake_images)
            dis_errG = dis_criterion(dis_output, real_labels)
            aux_errG = aux_criterion(aux_output, target)
            errG = dis_errG + aux_errG
            D_G_z2 = dis_output.mean().item()
            
            running_loss_G += errG
            running_loss_D += errD
        epoch_loss_G = running_loss_G / len(data_loader)
        epoch_loss_D = running_loss_D / len(data_loader)
    return epoch_loss_G, epoch_loss_D, D_x, D_G_z1, D_G_z2, aux_accuracy, real_images, fake_images
            
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