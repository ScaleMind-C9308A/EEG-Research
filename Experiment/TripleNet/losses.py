import torch 
import torch.nn as nn
import torch.nn.functional as F

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
        return torch.sum(eeg_embed*image_embed, dim=-1)
    def forward(self, anchor, positive, negative, average=True):
        """compute the similarity scores between anchor-positive and anchor-negative pairs"""
        pos_similarity = self.F(anchor, positive)
        neg_similarity = self.F(anchor, negative)
        # Triplet loss
        loss = F.relu(neg_similarity - pos_similarity + self.margin)
        return loss.mean() if average else loss.sum()

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
    def F(self, eeg_embed, image_embed):
        """Compability func F for compute the dot product between EEG and image representations"""
        return torch.sum(eeg_embed*image_embed, dim=-1)

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        pos_sim = self.F(embeddings[triplets[0]], embeddings[triplets[1]])
        neg_sim = self.F(embeddings[triplets[0]], embeddings[triplets[2]])
        losses = F.relu(neg_sim - pos_sim + self.margin)

        return losses.mean(), len(triplets)
