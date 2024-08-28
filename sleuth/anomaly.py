import argparse
import anndata as ad
from tqdm import tqdm
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from .read import load_pkl
from ._utils import seed_everything, update_configs_with_args
from .configs import AnomalyConfigs
from .model import GeneratorAD, Discriminator, Scorer


class AnomalyModel:
    def __init__(self, configs: AnomalyConfigs, anomaly_ratio: Optional[float]):
        # Number of epochs
        self.prepare_epochs = configs.prepare_epochs
        self.train_epochs = configs.train_epochs
        self.score_epochs = configs.score_epochs

        # Training
        self.batch_size = configs.batch_size
        self.learning_rate = configs.learning_rate
        self.n_critic = configs.n_critic
        self.loss_weight = configs.loss_weight
        self.device = configs.device

        # Initial model
        self._init_model(configs, anomaly_ratio)
        
        seed_everything(configs.random_state)
    
    def detect(self, ref: ad.AnnData):
        tqdm.write('Begin to train ACSleuth on the reference dataset...')

        self.gene_names = ref.var_names
        train_data = torch.Tensor(ref.X)
        self.loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True, drop_last=True)

        self.D.train()
        self.G.train()
        self._train(self.prepare_epochs, 'Preparation Epochs', False)
        self._train(self.train_epochs, 'Training Epochs', True)
        
        tqdm.write('Training has been finished.')

    def predict(self, tgt: ad.AnnData):
        self._check(tgt)

        tqdm.write('Begin to detect anomalies on the target dataset...')
        real_data = torch.Tensor(tgt.X)
        self.loader = DataLoader(real_data, batch_size=self.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True, drop_last=False)

        self.D.eval()
        self.G.eval()
        fake_data = []

        with torch.no_grad():
            for data in self.loader:
                data = data.to(self.device)
                fake, _ = self.G(data)
                fake_data.append(fake.detach())
  
        fake_data = torch.cat(fake_data, dim=0)
        delta = real_data.to(self.device) - fake_data

        self.S.train()
        with tqdm(total=self.score_epochs) as t:
            for _ in range(self.score_epochs):
                t.set_description(f'Prediction Epochs')

                _, loss = self.S(delta)
                self.opt_S.zero_grad()
                loss.backward()
                self.opt_S.step()
                self.sch_S.step()
                t.set_postfix(S_Loss = loss.item())
                t.update(1)

        self.S.eval()
        p = self.S.pred(delta)
        tqdm.write('Anomalies have been detected.')
        return p.cpu().detach().numpy().reshape(-1)

    def _init_model(self, configs: AnomalyConfigs, anomaly_ratio: Optional[float]):
        self.D = Discriminator(**configs.Discriminator).to(self.device)
        self.G = GeneratorAD(**configs.Generator).to(self.device)
        self.S = Scorer(anomaly_ratio=anomaly_ratio, **configs.Scorer).to(self.device)

        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.opt_S = optim.Adam(self.S.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))        

        self.sch_D = CosineAnnealingLR(self.opt_D, self.train_epochs)
        self.sch_G = CosineAnnealingLR(self.opt_G, self.train_epochs)
        self.sch_S = CosineAnnealingLR(self.opt_S, self.score_epochs)

        self.Loss = nn.L1Loss().to(self.device)

    def _train(self, epochs, description, train: bool):
        with tqdm(total=epochs) as t:
            for _ in range(epochs):
                t.set_description(description)

                for data in self.loader:
                    data = data.to(self.device)

                    for _ in range(self.n_critic):
                        self._UpdateD(data, train)

                    self._UpdateG(data, train)
        
                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)

                if train:
                    self.sch_D.step()
                    self.sch_G.step()

    def _UpdateD(self, data, train):
        if train:
            fake_data, _ = self.G(data)
        else:
            fake_data, _ = self.G.prepare(data)

        d1 = torch.mean(self.D(data))
        d2 = torch.mean(self.D(fake_data.detach()))
        gp = self.D.gradient_penalty(data, fake_data.detach())
        self.D_loss = - d1 + d2 + gp * self.weight['w_gp']

        self.opt_D.zero_grad()
        self.D_loss.backward()
        self.opt_D.step()
    
    def _UpdateG(self, data, train):
        if train:
            fake_data, z = self.G(data)
        else:
            fake_data, z = self.G.prepare(data)

        # discriminator provides feedback
        d = self.D(fake_data)

        L_rec = self.Loss(data, fake_data)
        L_adv = - torch.mean(d)
        self.G_loss = self.loss_weight['w_rec']*L_rec+self.loss_weight['w_adv']*L_adv
        self.opt_G.zero_grad()
        self.G_loss.backward()
        self.opt_G.step()

        self.G.Memory.update_mem(z)

    def _check(self, tgt: ad.AnnData):
        if (tgt.var_names != self.genes).any():
            raise RuntimeError('Target and reference data have different genes.')

        if (self.G is None or self.D is None):
            raise RuntimeError('Please train the model first.')


    def G_score(self, tgt: ad.AnnData):
        """
        Detect anomaly cells only with reconstruction errors from G.
        """
        self._check(tgt)

        real_data = torch.Tensor(tgt.X)
        self.loader = DataLoader(real_data, batch_size=self.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True, drop_last=False)

        self.D.eval()
        self.G.eval()
        fake_data = []
        
        with torch.no_grad():
            for data in self.loader:
                data = data.to(self.device)
                fake, _ = self.G.forward(data)
                fake_data.append(fake.detach())
  
        fake_data = torch.cat(fake_data, dim=0)
        delta = torch.norm(real_data.to(self.device) - fake_data, dim=1, p=2).reshape(-1)
        return delta.cpu().detach().numpy()
    
    def D_score(self, tgt: ad.AnnData):
        """
        Detect anomaly cells only with critic embeddings from D.
        """
        self._check(tgt)

        real_data = torch.Tensor(tgt.X)
        self.loader = DataLoader(real_data, batch_size=self.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True, drop_last=False)

        self.D.eval()
        self.G.eval()
        real_d, fake_d = [], []
        
        with torch.no_grad():
            for data in self.loader:
                data = data.to(self.device)
                fake, _ = self.G.forward(data)

                r_d = self.D.forward(data)
                real_d.append(r_d.detach())
                f_d = self.D.forward(fake)
                fake_d.append(f_d.detach())                
  
        real_d = torch.cat(real_d, dim=0)
        fake_d = torch.cat(fake_d, dim=0)
        delta = torch.norm(real_d - fake_d, dim=1, p=2).reshape(-1)
        return delta.cpu().detach().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ACSleuth for anomaly detection.')

    parser.add_argument('--data_path', type=str, help='Path to read the saved dataset.')
    parser.add_argument('--gene_dim', type=int, default=3000, help='Path to read the saved dataset.')
    parser.add_argument('--prepare_epochs', type=int, help='Epochs of preparing stage.')
    parser.add_argument('--train_epochs', type=int, help='Epochs of training stage.')
    parser.add_argument('--score_epochs', type=int, help='Epochs of updating scorer.')
    parser.add_argument('--batch_size', type=int, help='Batch size for training model.')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training model.')
    parser.add_argument('--n_critic', type=int, help='Train discriminator for n_critic times as every generator trained.')
    parser.add_argument('--loss_weight', type=Dict[str, float], help='Loss weight for training stage.')
    parser.add_argument('--random_state', type=Dict[str, float], help='Set random seed.')

    args = parser.parse_args()
    args_dict = vars(args)

    # Load dataset
    ref, tgt = load_pkl(args_dict.data_path)

    # update configs
    configs = AnomalyConfigs(args_dict.gene_dim)
    update_configs_with_args(configs, args_dict)

    model = AnomalyModel(configs)
    model.detect(ref)
    score = model.predict(tgt)