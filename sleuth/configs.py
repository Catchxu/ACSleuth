from ._utils import select_device


class AnomalyConfigs(object):
    def __init__(self, gene_dim):
        self.gene_dim = gene_dim

        # Number of epochs
        self.prepare_epochs = 50
        self.train_epochs = 20
        self.score_epochs = 20

        # Training
        self.batch_size = 128
        self.learning_rate = 1e-4
        self.n_critic = 2
        self.loss_weight = {'w_rec': 30, 'w_adv': 1, 'w_gp': 10}
        self.device = select_device('cuda:0')
