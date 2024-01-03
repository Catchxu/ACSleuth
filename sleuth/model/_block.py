import torch
from torch import nn
import math
from math import sqrt
from torch.nn import functional as F


class MemoryBlock(nn.Module):
    def __init__(self, mem_dim, z_dim, shrink_thres=0.005, temperature=0.5):
        """
        Initialize the MemoryBlock.

        Parameters
        ----------
        mem_dim : int
            Dimension of the memory.
        z_dim : int
            Dimension of the latent representation.
        shrink_thres : float, optional
            Threshold for the hard shrinkage operation.
        temperature : float, optional
            Temperature for the softmax operation.
        """
        super().__init__()
        self.mem_dim = mem_dim
        self.z_dim = z_dim
        self.shrink_thres = shrink_thres
        self.temperature = temperature
        self.register_buffer("mem", torch.randn(self.mem_dim, self.z_dim))
        self.register_buffer("mem_ptr", torch.zeros(1, dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset memory parameters with uniform initialization.
        """
        stdv = 1. / math.sqrt(self.mem.size(1))
        self.mem.data.uniform_(-stdv, stdv)

    @torch.no_grad()
    def update_mem(self, z):
        """
        Update the memory with a new latent representation.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation to be added to the memory.
        """
        batch_size = z.shape[0]  # z, B x C
        ptr = int(self.mem_ptr)
        assert self.mem_dim % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.mem[ptr:ptr + batch_size, :] = z  # mem, M x C
        ptr = (ptr + batch_size) % self.mem_dim  # move pointer

        self.mem_ptr[0] = ptr

    def hard_shrink_relu(self, x, lambd=0, epsilon=1e-12):
        """
        Hard shrinkage operation with ReLU.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        lambd : float, optional
            Threshold for the hard shrinkage.
        epsilon : float, optional
            Small constant to prevent division by zero.

        Returns
        -------
        x : torch.Tensor
            Output tensor after hard shrinkage.
        """
        x = (F.relu(x-lambd) * x) / (torch.abs(x - lambd) + epsilon)
        return x

    def forward(self, x):
        """
        Forward pass of the memory block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        output : torch.Tensor
            Output tensor after memory processing.
        """
        att_weight = torch.mm(x, self.mem.T)
        att_weight = F.softmax(att_weight/self.temperature, dim=1)

        # ReLU based shrinkage, hard shrinkage for positive value
        if (self.shrink_thres > 0):
            att_weight = self.hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        output = torch.mm(att_weight, self.mem)
        return output


class StyleBlock(nn.Module):
    def __init__(self, n_batch: int, z_dim: int):
        """
        Initialize the StyleBlock.

        Parameters
        ----------
        n_batch : int
            Number of batch in target dataset.
        z_dim : int
            Dimension of the latent representation.
        """
        super().__init__()
        self.n = n_batch
        self.style = nn.Parameter(torch.Tensor(n_batch, z_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset style parameters with uniform initialization.
        """
        stdv = 1. / math.sqrt(self.style.size(1))
        self.style.data.uniform_(-stdv, stdv)

    def forward(self, z, batchid):
        """
        Forward pass of the style block.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation.
        batchid : torch.Tensor
            Batch IDs.

        Returns
        -------
        output : torch.Tensor
            Output tensor after style transfering.
        """
        if self.n == 1:
            return z - self.style
        else:
            s = torch.mm(batchid, self.style)
            return z - s