import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

from .block import SNLinearBlock, ResNetBlock


class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim=[512, 64], num_blocks=1):
        super().__init__()

        # Discriminator layers
        disc_layers = []
        layers = [in_dim] + hidden_dim
        dim_1 = layers[0]
        for dim_2 in layers[1:]:
            disc_layers.append(SNLinearBlock(dim_1, dim_2))
            dim_1 = dim_2
        disc_layers.append(
            nn.Sequential(*[ResNetBlock(dim_2, spectral_norm=True) for _ in range(num_blocks)])
        )
        self.disc = nn.Sequential(*disc_layers)

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
    
        # Additional initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def gradient_penalty(self, real_data, fake_data):
        shapes = [1 if i != 0 else real_data.size(i) for i in range(real_data.dim())]
        cuda = True if torch.cuda.is_available() else False

        eta = torch.FloatTensor(*shapes).uniform_(0, 1)
        eta = eta.cuda() if cuda else eta
        interpolated = eta * real_data + ((1 - eta) * fake_data)
        interpolated = interpolated.cuda() if cuda else interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.forward(interpolated)

        # calculate gradients of probabilities with respect to examples
        grad = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                             grad_outputs=torch.ones(prob_interpolated.size()).cuda()
                             if cuda else torch.ones(prob_interpolated.size()),
                             create_graph=True, retain_graph=True)[0]

        grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def forward(self, x):
        x = self.disc(x)
        return x