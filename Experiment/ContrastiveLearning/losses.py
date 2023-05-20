import torch 
import torch.nn as nn
import torch.nn.functional as F

### TripletLoss takes in (anchor, positive, negative) embeddings and compute directly the loss value
### OnlineTripletLoss takes in (embeddings, labels) 
# => Use a triplet_selector to mine hard negative triplets
# -> Then finally it will calculate the loss based on the formed triplets

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample

    *margin=0 in the paper
    """
    def __init__(self, margin=0):
        super().__init__()   
        self.margin = margin
    def F(self, eeg_embed, image_embed):
        """Compability func F for compute the dot product between EEG and image representations"""
        # print(f'eeg embed size: ', eeg_embed.size())
        # print(f'img embed size: ', image_embed.size())
        return torch.sum(eeg_embed*image_embed, dim=-1)
    def forward(self, anchor, positive, negative, average=True):
        """compute the similarity scores between anchor-positive and anchor-negative pairs"""
        pos_similarity = self.F(anchor, positive)
        neg_similarity = self.F(anchor, negative)
        # Triplet loss
        loss = F.relu(neg_similarity - pos_similarity + self.margin)
        print(f"pos_sim: {pos_similarity.size()}, neg_sim: {neg_similarity.size()}, losses: {loss.size()}")
        print(f"pos_sim: {pos_similarity}, neg_sim: {neg_similarity}, losses: {loss}")
        return loss.mean() if average else loss.sum()

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, device, triplet_selector):
        """
        device: 
        """
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.device = device
        self.triplet_selector = triplet_selector
    def F(self, eeg_embeds, image_embeds):
        """
        Compability func F for compute the dot product between EEG and image representations
        Input:
            - eeg_embeds: (num_samples, eeg_dim) => tensor
            - img_embeds: (num_samples, img_dim) => tensor
            Assume eeg_dim == img_dim
        Return:
            - (num_samples, )
        """
        return torch.sum(eeg_embeds*image_embeds, dim=-1)

    def forward(self, eeg_embeddings, img_embeddings, target):
        """
        Input:
            - eeg_embeds: (num_samples, eeg_dim) => tensor
            - img_embeds: (num_samples, img_dim) => tensor
            - target: (num_samples,)
        
        triplets: (num_triplets, 3) => num_triplets < num_samples
        """
        triplets = self.triplet_selector.get_triplets(eeg_embeddings, img_embeddings, target)
        # print(f"Triplet size: {triplets.size()}")
        triplets.to(self.device)

        pos_sim = self.F(eeg_embeddings[triplets[:, 0]], img_embeddings[triplets[:, 1]])
        neg_sim = self.F(eeg_embeddings[triplets[:, 0]], img_embeddings[triplets[:, 2]])
        losses = F.relu( neg_sim -  pos_sim + self.margin)
        # losses = neg_sim - pos_sim + self.margin
        print(f"pos_sim: {pos_sim}, neg_sim: {neg_sim}, losses: {losses}")

        return losses.mean(), len(triplets)
