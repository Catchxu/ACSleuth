import anndata as ad
from tqdm import tqdm
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from ._utils import seed_everything
from .model import Discriminator, GeneratorDA
from .configs import AdaptConfigs


class PairDataset(Dataset):
    def __init__(self, ref_data, tgt_data):
        self.ref_data = ref_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.ref_data)

    def __getitem__(self, index):
        ref_sample = self.ref_data[index]
        tgt_sample = self.tgt_data[index]

        return {'ref': ref_sample, 'tgt': tgt_sample}


class AdaptModel:
    def __init__(self, configs: AdaptConfigs, num_batches: int):
        # Training
        self.n_epochs = configs.n_epochs
        self.batch_size = configs.batch_size
        self.learning_rate = configs.learning_rate
        self.n_critic = configs.n_critic
        self.loss_weight = configs.loss_weight
        self.device = configs.device
        
        # Initial model
        self._init_model(configs, num_batches)

        seed_everything(configs.random_state)
    
    def _init_model(self, configs: AdaptConfigs, num_batches: int):
        self.D = Discriminator(**configs.Discriminator).to(self.device)
        self.G = GeneratorDA(num_batches, **configs.Generator).to(self.device)

        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))     

        self.sch_D = CosineAnnealingLR(self.opt_D, self.n_epochs)
        self.sch_G = CosineAnnealingLR(self.opt_G, self.n_epochs)

        self.Loss = nn.L1Loss().to(self.device)        