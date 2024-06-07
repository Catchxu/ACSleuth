import anndata as ad
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from ._utils import seed_everything
from .model import Discriminator, GeneratorDA, GeneratorAD
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
    
    def adapt(self, ref: ad.AnnData, tgt: ad.AnnData, batch_key: str, generator: GeneratorAD):
        self._check(ref, tgt, batch_key)
        tqdm.write('Begin to correct data domain shifts...')
        ref_data, tgt_data = self._map(ref, tgt, generator)

        dataset = PairDataset(ref_data, tgt_data)
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True, drop_last=False)

        self.D.train()
        self.G.train()
        self._train(self.n_epochs)

        # Generate data without domain shifts
        dataset = torch.Tensor(tgt.X)
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True, drop_last=False)

        self.G.eval()
        corrected = []
        with torch.no_grad():
            for data in self.loader:
                data = data.to(self.device)
                fake_ref = self.G(data)
                corrected.append(fake_ref.cpu().detach())

        corrected = torch.cat(corrected, dim=0).numpy()
        tgt.X = corrected
        tqdm.write('Data domain shifts have been corrected.')
        return tgt

    def _init_model(self, configs: AdaptConfigs, num_batches: int):
        self.D = Discriminator(**configs.Discriminator).to(self.device)
        self.G = GeneratorDA(num_batches, **configs.Generator).to(self.device)

        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))     

        self.sch_D = CosineAnnealingLR(self.opt_D, self.n_epochs)
        self.sch_G = CosineAnnealingLR(self.opt_G, self.n_epochs)

        self.Loss = nn.L1Loss().to(self.device)

    def _check(self, ref, tgt, batch_key):
        if (tgt.var_names != ref.var_names).any():
            raise RuntimeError('Target and reference data have different genes!')

        if batch_key not in tgt.obs.columns:
            raise RuntimeError(f'{batch_key} is not in tgt.obs!')

    @torch.no_grad()
    def _map(self, ref: ad.AnnData, tgt: ad.AnnData, generator: GeneratorAD):
        ref_data = torch.Tensor(ref.X).to(self.device)
        tgt_data = torch.Tensor(tgt.X).to(self.device)

        generator.eval()
        ref_e = generator(ref_data)
        tgt_e = generator(tgt_data)

        dot_product_matrix = torch.mm(tgt_e, ref_e.t())
        max_indices = torch.argmax(dot_product_matrix, dim=1)
        mapped_ref_e = ref_e[max_indices]
        return mapped_ref_e.detach().cpu(), tgt_data.cpu()
    
    def _train(self, epochs):
        with tqdm(total=epochs) as t:
            for _ in range(epochs):
                t.set_description('Adaptation Epochs')

                for data in self.loader:
                    ref = data['ref'].to(self.device)
                    tgt = data['tgt'].to(self.device)

                    for _ in range(self.n_critic):
                        self._UpdateD(ref, tgt)

                    self._UpdateG(ref, tgt)
        
                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)
                self.sch_D.step()
                self.sch_G.step()
    
    def _UpdateD(self, ref, tgt):
        fake_ref = self.G(tgt)

        d1 = torch.mean(self.D(ref))
        d2 = torch.mean(self.D(fake_ref.detach()))
        gp = self.D.gradient_penalty(ref, fake_ref.detach())
        self.D_loss = - d1 + d2 + gp * self.weight['w_gp']

        self.opt_D.zero_grad()
        self.D_loss.backward()
        self.opt_D.step()
    
    def _UpdateG(self, ref, tgt):
        fake_ref = self.G(tgt)

        # discriminator provides feedback
        d = self.D(fake_ref)

        L_rec = self.Loss(ref, fake_ref)
        L_adv = - torch.mean(d)
        self.G_loss = self.loss_weight['w_rec']*L_rec+self.loss_weight['w_adv']*L_adv
        self.opt_G.zero_grad()
        self.G_loss.backward()
        self.opt_G.step()