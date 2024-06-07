import anndata as ad
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from ._utils import seed_everything
from .model import GeneratorAD, Cluster
from .configs import SubtypeConfigs


class PairedDataset(Dataset):
    def __init__(self, z, res):
        self.z = z
        self.res = res

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.res[idx]


class SubtypeModel:
    def __init__(self, configs: SubtypeConfigs, generator: GeneratorAD, num_types: int):
        self.n_epochs = configs.n_epochs
        self.batch_size = configs.batch_size
        self.learning_rate = configs.learning_rate
        self.weight_decay = configs.weight_decay
        self.device = configs.device

        # Trained generator
        self.G = generator

        # Initial model
        self._init_model(configs, generator, num_types)

        seed_everything(configs.random_state)
    
    def _init_model(self, configs: SubtypeConfigs, generator: GeneratorAD, num_types: int):
        self.C = Cluster(generator, num_types, **configs.Cluster).to(self.device)
        self.opt_C = optim.Adam(self.C.parameters(), 
                                lr=self.learning_rate, 
                                betas=(0.5, 0.999),
                                weight_decay=self.weight_decay)
        self.sch_C = CosineAnnealingLR(self.opt_C, self.n_epochs)

    def fit(self, adata: ad.AnnData):
        data = torch.Tensor(adata.X).to(self.device)
        z, res = self.generate_z_res(data)

        self.C.mu_init(z.cpu().detach().numpy())
        dataset = PairedDataset(z, res)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


        self.C.train()
        self._train(self.n_epochs)

        with torch.no_grad():
            self.C.eval()
            _, q = self.C.forward(z, res)
            return q

    @torch.no_grad()
    def generate_z_res(self, data: torch.Tensor):
        fake, z = self.G(data)
        res = data - fake.detach()
        return z, res
    
    def _train(self, epochs):
        with tqdm(total=epochs) as t:
            for _ in range(epochs):
                t.set_description('CLustering Epochs')

                for batch_z, batch_res in self.dataloader:
                    _, q = self.C(batch_z, batch_res)
                    p = self.C.target_distribution(q).data

                    self.opt_C.zero_grad()
                    Loss = self.C.loss_function(p, q)
                    Loss.backward(retain_graph=True)
                    self.opt_C.step()
                
                t.set_postfix(Loss = Loss.item())
                t.update(1)
                self.sch_C.step()

