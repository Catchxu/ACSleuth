import anndata as ad
from tqdm import tqdm
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ._utils import seed_everything
from .model import GeneratorAD, Discriminator, Predictor, GeneratorDA





class FineSleuth:
    def __init__(self,
                 adapt_epochs: int = 50,
                 clust_epochs: int = 60,
                 batch_size: int = 256,
                 learning_rate: float = 3e-4,
                 n_critic: int = 5,
                 GPU: bool = True,
                 weight: Optional[Dict[str, float]] = None,
                 random_state: Optional[int] = None):
        """
        Initializes the FineSleuth class with hyperparameters and optional settings.

        Parameters
        ----------
        adapt_epochs : int, optional
            Number of epochs for the data adaptation phase.
        clust_epochs : int, optional
            Number of epochs for the clustering phase.
        batch_size : int, optional
            Batch size for DataLoader in data adaptation phase.
        learning_rate : float, optional
            Initial learning rate for the optimizer.
        n_critic : int, optional
            Number of discriminator updates per generator update.
        GPU : bool, optional
            If True, uses GPU if available; otherwise, uses CPU.
        weight : dict, optional
            Dictionary containing weights for losses.
        random_state : int, optional
            Seed for reproducibility.
        """
        self.adapt_epochs = adapt_epochs
        self.clust_epochs = clust_epochs
        self.bs = batch_size
        self.lr = learning_rate
        self.n_critic = n_critic
        self.device = torch.device("cuda:0" if GPU and torch.cuda.is_available() else "cpu")
        self.weight = weight or {'w_rec': 50, 'w_adv': 1, 'w_gp': 10}
        
        if random_state is not None:
            seed_everything(random_state)
    
    def _check(self, ref, tgt, batch_key):
        """
        Check if the target dataset is compatible with the reference dataset.
        Check if the target dataset has batch id.

        Parameters
        ----------
        ref : anndata.AnnData
            Reference dataset.
        tgt : anndata.AnnData
            Target dataset.
        batch_key: str
            Batch ID key in tgt.obs for batch information.

        Raises
        ------
        RuntimeError
            If genes are different or if the model is not trained.
        RuntimeError
            batch_key can't find in the tgt.obs.
        """
        if (tgt.var_names != ref.var_names).any():
            raise RuntimeError('Target and reference data have different genes!')

        if batch_key not in tgt.obs.columns:
            raise RuntimeError(f'{batch_key} is not in tgt.obs!')

    @torch.no_grad()
    def _map(self, ref, tgt, encoder):
        ref_data = torch.Tensor(ref.X).to(self.device)
        tgt_data = torch.Tensor(tgt.X).to(self.device)

        encoder.eval()
        ref_e = encoder(ref_data)
        tgt_e = encoder(tgt_data)

        dot_product_matrix = torch.mm(tgt_e, ref_e.t())
        max_indices = torch.argmax(dot_product_matrix, dim=1)
        mapped_ref_e = ref_e[max_indices]
        return mapped_ref_e.detach().cpu(), tgt_data.cpu()

    def _create_opt_sch(self, model, lr, T_max=None):
        """
        Creates optimizer and scheduler for a given model.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model.
        lr : float
            Learning rate.
        T_max : int, optional
            Maximum number of iterations for the scheduler.

        Returns
        -------
        optimizer : torch.optim.Optimizer
            PyTorch optimizer.
        scheduler : torch.optim.lr_scheduler._LRScheduler or None
            PyTorch scheduler (None if not provided).
        """
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=T_max) if T_max else None
        return optimizer, scheduler

    def _train(self, epochs):
        """
        Training loop for the FineSleuth data adaptation model.

        Parameters
        ----------
        epochs : int
            Number of epochs.
        """
        with tqdm(total=epochs) as t:
            for _ in range(epochs):
                t.set_description('Data Adaptation Epochs')

                for data in self.loader:
                    ref = data['ref'].to(self.device)
                    tgt = data['tgt'].to(self.device)

                    for _ in range(self.n_critic):
                        self._update_D(ref, tgt)

                    self._update_G(ref, tgt)
        
                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)
                self.sch_D.step()
                self.sch_G.step()

    def _update_D(self, ref, tgt):
        """
        Update the discriminator during training.

        Parameters
        ----------
        ref : torch.Tensor
            Reference data.
        tgt : torch.Tensor
            Target data.
        """
        fake_ref = self.G.forward(tgt)

        d1 = torch.mean(self.D.forward(ref))
        d2 = torch.mean(self.D.forward(fake_ref.detach()))
        gp = self.D.gradient_penalty(ref, fake_ref.detach())
        self.D_loss = - d1 + d2 + gp * self.weight['w_gp']

        self.opt_D.zero_grad()
        self.D_loss.backward()
        self.opt_D.step()
    
    def _update_G(self, ref, tgt):
        """
        Update the generator during training.

        Parameters
        ----------
        ref : torch.Tensor
            Reference data.
        tgt : torch.Tensor
            Target data.
        """
        fake_ref = self.G.forward(tgt)

        # discriminator provides feedback
        d = self.D.forward(fake_ref)

        L_rec = self.L1(ref, fake_ref)
        L_adv = -torch.mean(d)
        self.G_loss = self.weight['w_rec']*L_rec + self.weight['w_adv']*L_adv
        self.opt_G.zero_grad()
        self.G_loss.backward()
        self.opt_G.step()

    def data_adapt(self, ref: ad.AnnData, tgt: ad.AnnData, batch_key: str,
                   encoder: nn.Module):
        self._check(ref, tgt, batch_key)

        ref_data, tgt_data = self._map(ref, tgt, encoder)
        dataset = PairDataset(ref_data, tgt_data)
        self.loader = DataLoader(dataset, batch_size=self.bs, shuffle=True,
                                 num_workers=4, pin_memory=True, drop_last=False)

        n_batch = len(tgt.obs[batch_key].unique())
        self.D = Discriminator(ref.n_vars).to(self.device)
        self.G = GeneratorDA(n_batch, ref.n_vars).to(self.device) 
        self.opt_D, self.sch_D = self._create_opt_sch(self.D, self.lr, self.adapt_epochs)
        self.opt_G, self.sch_G = self._create_opt_sch(self.G, self.lr, self.adapt_epochs)
        self.L1 = nn.L1Loss().to(self.device)

        self.D.train()
        self.G.train()
        self._train(self.adapt_epochs)





        
