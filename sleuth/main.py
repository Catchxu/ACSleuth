import anndata as ad
from tqdm import tqdm
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ._utils import seed_everything
from .model import GeneratorAD, Discriminator


class CoarseSleuth:
    def __init__(self, 
                 n_epochs: int = 100,
                 batch_size: int = 64,
                 learning_rate: float = 2e-4,
                 n_critic: int = 1, 
                 GPU: bool = True,
                 weight: Optional[Dict[str, float]] = None,
                 random_state: Optional[int] = None,
                 **kwargs):
        self.n_epochs = n_epochs
        self.bs= batch_size
        self.lr = learning_rate
        self.n_critic = n_critic

        if GPU:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                print("GPU isn't available, and use CPU to train ODBC-GAN.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if weight is None:
            self.weight = {'w_rec': 50, 'w_adv': 1, 'w_gp': 10}
        else:
            self.weight = weight
        
        if random_state is not None:
            seed_everything(random_state)
        
        self.kwargs = kwargs
    
    def detector(self, ref: ad.AnnData, prepare_epochs: 20):
        tqdm.write('Begin to train ACsleuth on the reference dataset...')

        self.genes = ref.var_names
        train_data = torch.Tensor(ref.X)
        self.loader = DataLoader(train_data, batch_size=self.bs, shuffle=True,
                                 num_workers=2, pin_memory=True, drop_last=True)

        self.D = Discriminator(ref.n_vars).to(self.device)
        self.G = GeneratorAD(ref.n_vars, **self.kwargs).to(self.device)

        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.sch_D = optim.lr_scheduler.CosineAnnealingLR(optimizer = self.opt_D,
                                                          T_max = self.n_epochs)
        self.sch_G = optim.lr_scheduler.CosineAnnealingLR(optimizer = self.opt_D,
                                                          T_max = self.n_epochs)
        self.L1 = nn.L1Loss().to(self.device)

        self.D.train()
        self.G.train()

        self._prepare(prepare_epochs)

        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description(f'Train Epochs')

                for _, data in enumerate(self.loader):
                    data = data.to(self.device)
                    self._train(prepare=False)

                # Update learning rate for G and D
                self.sch_D.step()
                self.sch_G.step()
                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)
        
        tqdm.write('Training has been finished.')

    def _prepare(self, prepare_epochs):
        with tqdm(total=prepare_epochs) as t:
            for _ in range(prepare_epochs):
                t.set_description(f'Prepare Epochs')

                for _, data in enumerate(self.loader):
                    data = data.to(self.device)
                    self._train(prepare=True)

                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)
    
    def _train(self, prepare):
        for _, data in enumerate(self.loader):
                data = data.to(self.device)

                for _ in range(self.n_critic):
                    self._update_D(data, prepare)
                
                self._update_G(data, prepare)
    
    def _update_D(self, data, prepare):
        '''Updating discriminator'''
        self.opt_D.zero_grad()
        fake_data = self.G.prepare(data) if prepare else self.G(data)

        d1 = torch.mean(self.D(data))
        d2 = torch.mean(self.D(fake_data.detach()))
        gp = self.D.gradient_penalty(data, fake_data.detach())

        self.D_loss = - d1 + d2 + gp * self.weight['w_gp']
        self.D_loss.backward()
        self.opt_D.step()
    
    def _update_G(self, data, prepare):
        '''Updating generator'''
        self.opt_G.zero_grad()
        fake_data = self.G.prepare(data) if prepare else self.G(data)
        
        # discriminator provides feedback
        d = self.D(fake_data)

        L_rec = self.L1(data, fake_data)
        L_adv = -torch.mean(d)
        self.G_loss = self.weight['w_rec']*L_rec + self.weight['w_adv']*L_adv
        self.G_loss.backward()
        self.opt_G.step()







        
