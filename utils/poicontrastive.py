import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from sklearn import preprocessing
import argparse
import math


class POIContrastiveLoss(nn.Module):
    def __init__(self, temperature=10, contrast_mode='one'):
        super(POIContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, real):
        device = torch.device('cuda:0')
        batch_size = features.shape[0]
        features = torch.squeeze(features)
        real = torch.squeeze(real)
        anchor_sample = features[0, :]
        
        dist_matrix = torch.norm(real[:, None] - real, dim=2, p=2)
        sim_function = nn.PairwiseDistance(p=1)
        closest_id = 0
        if self.contrast_mode == 'one':
            closest_id = np.argmin(dist_matrix[0, 1:].detach().cpu().numpy()) + 1

            negative_sum = 0          
            for i in range(1, batch_size):
                if i != closest_id:
                    one_term = torch.exp(sim_function(anchor_sample, features[i, :])/ self.temperature).to(device)
                    negative_sum += one_term            
                    positive_sum = torch.exp(sim_function(anchor_sample, features[closest_id, :])/ self.temperature).to(device)
            loss = torch.neg(torch.log(positive_sum / negative_sum)).to(device)
            return loss
        elif self.contrast_mode == 'all':
            sum_loss = 0
            for i in range(1, batch_size):
                negative_sum = 0
                for j in range(1, batch_size):
                    if j != i and dist_matrix[0, j] > dist_matrix[0, i]:
                        one_term = torch.exp(sim_function(anchor_sample, features[j, :])/ self.temperature).to(device)
                        negative_sum += one_term
                positive_sum = torch.exp(sim_function(anchor_sample, features[i, :])/ self.temperature).to(device)
                sum_loss += torch.log(positive_sum / negative_sum).to(device)
            return sum_loss
        else:
            print('Wrong Contrastive Mode!')            