import torch.nn as nn
import torch.nn.utils.spectral_norm as SNorm


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim,
                 norm: bool = True, act: bool = True, dropout: bool = True):
        super().__init__()
        self.linear = nn.Sequential(
            SNorm(nn.Linear(in_dim, out_dim))
            if norm else nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
            nn.Dropout(0.1) if dropout else nn.Identity(),
        )

    def forward(self, x):
        return self.linear(x)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            LinearBlock(dim, dim),
            LinearBlock(dim, dim, False, False, False)
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.act(x + self.fc(x))
    

class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim=[512, 256, 16], n_Res=2):
        super().__init__()

        self.disc_list = nn.ModuleList()
        layers = [in_dim] + hidden_dim
        dim_1 = layers[0]
        for dim_2 in layers[1:]:
            self.disc_list.append(
                LinearBlock(dim_1, dim_2)
            )
            dim_1 = dim_2
        self.disc_list.append(
            nn.Sequential(*[ResBlock[dim_2, dim_2] for _ in range(n_Res)])
        )

    def forward(self, x):
        for layer in self.disc_list:
            x = layer(x)
        return x