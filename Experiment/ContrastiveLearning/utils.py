import numpy as np
import torch
from itertools import combinations

def similarity_matrix(eeg_embeds, img_embeds):
    """Compute the similarity matrix using the compatibility function
    Input:
        - eeg_embeds: (eeg_num_samples, eeg_dim) => tensor
        - img_embeds: (img_num_samples, img_dim) => tensor
        Assume eeg_dim == img_dim
    return:
        - sim_matrix: (eeg_num_samples, img_num_samples)
    """
    # sim_matrix = torch.zeros(eeg_embeds.size()[0], img_embeds.size()[0])
    # for i, eeg in enumerate(eeg_embeds):
    #     for j, image in enumerate(img_embeds):
    #         sim_matrix[i, j] = torch.sum(eeg*image, dim=-1)
    sim_matrix = torch.matmul(eeg_embeds, img_embeds.t())
    return sim_matrix

class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples

    Return: np.array (N_triplets, 3)
    """
    def __init__(self):
        pass
    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, eeg_embeddings, img_embeddings, labels):
        # print(f"EEG embeddings: {eeg_embeddings.size()}")
        # print(f"Image embeddings: {img_embeddings.size()}")
        if self.cpu:
            eeg_embeddings = eeg_embeddings.cpu()
            img_embeddings = img_embeddings.cpu()
        sim_matrix = similarity_matrix(eeg_embeddings, img_embeddings)
        sim_matrix = sim_matrix.cpu()
        # print(sim_matrix)
        # print(f"sim_matrix size: {sim_matrix.size()}")

        labels = labels.cpu().data.numpy()
        # maximum total triplets found: n_labels*n_samples
        # => dim of triplets (maximum): (n_labels*n_samples, 3)
        triplets = [] 

        for label in set(labels):
            # loop for n_classes times
            #len(labels) == batch_size
            #len(eeg_indices) ==len(pos_img_indices) == n_samples
            #len(negatvie_indices) == batch_size - n_samples
            label_mask = (labels == label)
            eeg_indices = np.where(label_mask)[0] 
            pos_img_indices = eeg_indices
            # # We want each eeg pair with its corresponding image 
            # # => don't want to shuffle img_indices
            # pos_img_indices = np.random.permutation(eeg_indices)
            if len(eeg_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]


            pos_sims = sim_matrix[eeg_indices, pos_img_indices]
            #pos_sims: [sim1(eeg1, pos_img1),..., sim_n(eeg1, pos_img1)] => (n_samples,)
            # print(f"pos_sims: {pos_sims}")
            for anchor_idx, pos_idx, pos_sim in zip(eeg_indices, pos_img_indices, pos_sims):
                neg_sim = sim_matrix[torch.LongTensor(np.array([anchor_idx])), torch.LongTensor(negative_indices)]
                #important fix: neg_sim - pos_sim, not the other way!!
                loss_values = neg_sim - pos_sim  + self.margin 
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_idx,pos_idx, hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_idx, pos_idx, negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)