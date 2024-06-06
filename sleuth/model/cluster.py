import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from tqdm import tqdm
from sklearn.cluster import KMeans

from .generator import GeneratorAD
from .block import TFBlock


class Cluster(nn.Module):
    def __init__(self, 
                 generator: GeneratorAD, 
                 num_types: int,
                 alpha: float = 1,
                 KMeans_n_init: int = 20,
                 learning_rate: float = 1e-4,
                 n_epochs: int = 2000,
                 update_interval: int = 10,
                 weight_decay: float = 1e-4,
                 **kwargs
                 ):
        super().__init__()

        self.G = generator
        self.z_dim = self.G.z_dim
        self.num_types = num_types
        self.alpha = alpha
        self.KMeans_n_init = KMeans_n_init
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.update_interval = update_interval
        self.weight_decay = weight_decay

        self.fusion = TFBlock(self.z_dim, **kwargs)
        self.mu = Parameter(torch.Tensor(self.num_types, self.z_dim))

        # classifer for supervised pre-training
        self.classifer = nn.Linear(self.z_dim, num_types)

    def forward(self, z, res_z):
        z = self.fusion(z, res_z)
        q = 1.0/((1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2)/self.alpha)+1e-8)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        self.mu_update(z, q)
        return z, q

    def pretrain(self, z, res):
        res_z = self.G.Encoder(res)
        z = self.fusion(z, res_z)
        return self.classifer(z)

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def mu_init(self, feat):
        kmeans = KMeans(self.num_types, n_init=self.KMeans_n_init)
        y_pred = kmeans.fit_predict(feat)
        feat = pd.DataFrame(feat, index=np.arange(0, feat.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, feat.shape[0]), name="Group")
        Mergefeat = pd.concat([feat, Group], axis=1)
        centroid = np.asarray(Mergefeat.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(centroid))

    def mu_update(self, feat, q):
        y_pred = torch.argmax(q, axis=1).cpu().numpy()
        feat = pd.DataFrame(feat.cpu().detach().numpy(), index=np.arange(0, feat.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, feat.shape[0]), name='Group')
        Mergefeat = pd.concat([feat, Group], axis=1)
        centroid = np.asarray(Mergefeat.groupby('Group').mean())

        self.mu.data.copy_(torch.Tensor(centroid))

    def fit(self, z, res_z):
        # z and res_z are obtained by Fullforward, SCforward, or STforward
        optimizer = optim.Adam(self.parameters(), 
                               lr=self.learning_rate, 
                               weight_decay=self.weight_decay)

        z = self.fusion(z, res_z)
        self.mu_init(z.cpu().detach().numpy())

        self.train()
        with tqdm(total=self.n_epochs) as t:
            for epoch in range(self.n_epochs):
                t.set_description(f'Train Epochs')

                p, q = self.forward(z, res_z)

                if epoch % self.update_interval == 0:
                    p = self.target_distribution(q).data

                optimizer.zero_grad()
                loss = self.loss_function(p, q)
                loss.backward(retain_graph=True)
                optimizer.step()

                t.set_postfix(Loss = loss.item())
                t.update(1)

        with torch.no_grad():
            self.eval()
            _, q = self.forward(z, res_z)
            return q