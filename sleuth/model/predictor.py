import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import LinearBlock, ResBlock


class Predictor(nn.Module):
    def __init__(self, in_dim, n_Res=2, hidden_dim=[512, 256]):
        super().__init__()
        
        pred_layers = []
        layers = [in_dim] + hidden_dim
        dim_1 = layers[0]
        for dim_2 in layers[1:]:
            pred_layers.append(LinearBlock(dim_1, dim_2))
            dim_1 = dim_2
        pred_layers.append(nn.Sequential(*[ResBlock(dim_2) for _ in range(n_Res)]))
        
        # output layer
        pred_layers.append(LinearBlock(dim_2, 1))
        self.pred = nn.Sequential(*pred_layers)

        # Additional initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def _est_m_n(self, p):
        n = torch.sum(p)
        m = p.shape[0] - n
        return m, n

    def _gamma(self, p):
        m, n = self._est_m_n(p)
        pi_p = math.pi * p
        cof = torch.mm(pi_p, pi_p.t()) / (math.pi * math.pi)
        
        term1 = m * (m - 1) * torch.mm(p, p.t())
        term2 = m * n * torch.mm(p - 1, p.t())
        term3 = m * n * torch.mm(p, (p - 1).t())
        term4 = n * (n - 1) * torch.mm(p - 1, (p - 1).t())
        
        result = cof / (term1 - term2 - term3 + term4 + 1e-5)
        return result

    def _loss(self, delta, p):
        gamma = self._gamma(p)
        k = torch.mm(delta, delta.t())
        return torch.mean(k * gamma)

    def forward(self, delta):
        p = torch.sigmoid(self.pred(delta))
        loss = self._loss(delta, p)
        return p, loss
        
        