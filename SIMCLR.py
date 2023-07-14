import torch
import torch.nn as nn
import numpy as np


class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size #200
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):

        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) #/ self.temperature      #200x200
        positive_samples = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        #positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) #200x1
     
        #for i in range(self.batch_size):
          #pos = positive_samples[i][0].tolist()
          #arrPos.append(pos)  
        #tensPos = torch.tensor(arrPos)
        
        
        negative_samples = sim[self.mask].reshape(N, -1)  #200x198
        
        minNegSamples = torch.max(negative_samples, 1)
        minNegSamples = minNegSamples[0]
        minNegSamples = minNegSamples[:100]
        # print(minNegSamples.size())
        
        return positive_samples, minNegSamples