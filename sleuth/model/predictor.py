import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import LinearBlock, ResBlock


class Predictor(nn.Module):
    def __init__(self, in_dim, n_Res=2, hidden_dim=[16, 1], ):
        super().__init__()

        pred_layers = []
        layers = [in_dim] + hidden_dim
        dim_1 = layers[0]
        pred_layers.append(nn.Sequential(*[ResBlock[dim_1, dim_1] for _ in range(n_Res)]))
        for dim_2 in layers[1:]:
            pred_layers.append(LinearBlock(dim_1, dim_2))
            dim_1 = dim_2
        
        # output layer
        pred_layers.append(nn.Sigmoid())
        self.pred = nn.Sequential(*pred_layers)

        self.kernel_size = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # Additional initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def _kernel(self, x, y):
        D = F.pairwise_distance(x.unsqueeze(1), y.unsqueeze(0), p=2)
        kernel = torch.exp(-0.5 * (D / self.kernel_size)**2)
        return kernel
    
    def _MMD(self, real, fake):
        D1 = self._kernel(real, real)
        D2 = self._kernel(fake, fake)
        D3 = self._kernel(real, fake)
        return D1 + D2 - D3 - D3.T

    def forward(self, real, fake):
        p = self.pred(real - fake)
        A = 1 - p.view(-1, 1) - p.view(1, -1)
        MMD = self._MMD(real, fake)
        loss = torch.mean(A * MMD)
        return p, loss
        
        