import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
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
    def __init__(self, in_dim, hidden_dim=[512, 256, 64], n_Res=2):
        """
        Initialize the Discriminator.

        Parameters
        ----------
        in_dim : int
            Input dimension.
        hidden_dim : list of int, optional
            List of hidden layer dimensions.
        n_Res : int, optional
            Number of residual blocks.
        """
        super().__init__()

        # Discriminator layers
        disc_layers = []
        layers = [in_dim] + hidden_dim
        dim_1 = layers[0]
        for dim_2 in layers[1:]:
            disc_layers.append(LinearBlock(dim_1, dim_2))
            dim_1 = dim_2
        disc_layers.append(nn.Sequential(*[ResBlock(dim_2) for _ in range(n_Res)]))
        self.disc = nn.Sequential(*disc_layers)

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
    
        # Additional initialization
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights using Xavier normal initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def gradient_penalty(self, real_data, fake_data):
        """
        Calculate gradient penalty for training discriminator.

        Parameters
        ----------
        real_data : torch.Tensor
            Real data.
        fake_data : torch.Tensor
            Fake data.

        Returns
        -------
        grad_penalty : torch.Tensor
            Gradient penalty.
        """
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
        """
        Forward pass of the discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        x : torch.Tensor
            Output data.
        """
        x = self.disc(x)
        return x