import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as SNorm
import math
from torch.nn import functional as F



class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim,
                 norm = True, act = True, dropout = True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
            nn.Dropout(0.1) if dropout else nn.Identity()
        )

    def forward(self, x):
        return self.layer(x)




class SNLinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim,
                 norm = True, act = True, dropout = True):
        super().__init__()
        self.layer = nn.Sequential(
            SNorm(nn.Linear(in_dim, out_dim))
            if norm else nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
            nn.Dropout(0.1) if dropout else nn.Identity()
        )

    def forward(self, x):
        return self.layer(x)




class ResNetBlock(nn.Module):
    def __init__(self, dim, spectral_norm: False):
        super().__init__()
        if spectral_norm:
            self.fc = nn.Sequential(
                SNLinearBlock(dim, dim),
                SNLinearBlock(dim, dim, False, False, False)
            )
        else:
            self.fc = nn.Sequential(
                LinearBlock(dim, dim),
                LinearBlock(dim, dim, False, False, False)
            )

        self.act = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.act(x + self.fc(x))




class Extractor(nn.Module):
    def __init__(self, in_dim, hidden_dim=[512, 256], num_blocks=2):
        super().__init__()

        # Encoder layers
        encoder_layers = []
        layers = [in_dim] + hidden_dim

        dim_1 = layers[0]
        for dim_2 in layers[1:]:
            encoder_layers.append(LinearBlock(dim_1, dim_2))
            dim_1 = dim_2
        encoder_layers.append(
            nn.Sequential(*[ResNetBlock(dim_2) for _ in range(num_blocks)])
        )
        self.Encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        decoder_layers = []
        layers = layers[::-1]

        dim_1 = layers[0]
        decoder_layers.append(
            nn.Sequential(*[ResNetBlock(dim_1) for _ in range(num_blocks)])
        )
        for dim_2 in layers[1:]:
            if dim_2 != layers[-1]:
                decoder_layers.append(LinearBlock(dim_1, dim_2))
            else:
                # the last layer don't have norm, act & dropout
                decoder_layers.append(LinearBlock(dim_1, dim_2, False, False, False))
            dim_1 = dim_2

        decoder_layers.append(nn.ReLU())
        self.Decoder = nn.Sequential(*decoder_layers)

        # Additional initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)





class MemoryBlock(nn.Module):
    def __init__(self, mem_dim, z_dim, shrink_thres=0.005, temperature=0.5):
        super().__init__()
        self.mem_dim = mem_dim
        self.z_dim = z_dim
        self.shrink_thres = shrink_thres
        self.temperature = temperature
        self.mem = torch.randn(self.mem_dim, self.z_dim)
        self.mem_ptr = torch.zeros(1, dtype=torch.long)

        self._init_parameters()

    def _init_parameters(self):
        stdv = 1. / math.sqrt(self.mem.size(1))
        self.mem.data.uniform_(-stdv, stdv)

    @torch.no_grad()
    def update_mem(self, z):
        batch_size = z.shape[0]  # z, B x C
        ptr = int(self.mem_ptr)
        assert self.mem_dim % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.mem[ptr:ptr + batch_size, :] = z  # mem, M x C
        ptr = (ptr + batch_size) % self.mem_dim  # move pointer

        self.mem_ptr[0] = ptr

    def hard_shrink_relu(self, x, lambd=0, epsilon=1e-12):
        x = (F.relu(x-lambd) * x) / (torch.abs(x - lambd) + epsilon)
        return x

    def forward(self, x):
        att_weight = torch.mm(x, self.mem.T)
        att_weight = F.softmax(att_weight/self.temperature, dim=1)

        # ReLU based shrinkage, hard shrinkage for positive value
        if (self.shrink_thres > 0):
            att_weight = self.hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        output = torch.mm(att_weight, self.mem)
        return output




class StyleBlock(nn.Module):
    def __init__(self, num_batches: int, z_dim: int):
        super().__init__()
        self.n = num_batches
        self.style = nn.Parameter(torch.Tensor(num_batches, z_dim))
        self._init_parameters()

    def _init_parameters(self):
        stdv = 1. / math.sqrt(self.style.size(1))
        self.style.data.uniform_(-stdv, stdv)

    def forward(self, z, batchid):
        if self.n == 1:
            return z - self.style
        else:
            s = torch.mm(batchid, self.style)
            return z - s