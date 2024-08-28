from ._utils import select_device


class AnomalyConfigs:
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
        self.device = select_device('cuda')
        self.random_state = 2024

        # model
        self.Discriminator = {
            'in_dim': self.gene_dim,
            'hidden_dim': [256, 16],
            'num_blocks': 1
        }

        self.Generator = {
            'in_dim': self.gene_dim,
            'hidden_dim': [512, 256],
            'num_blocks': 2,
            'mem_dim': 512,
            'threshold': 0.01,
            'temperature': 0.05      
        }

        self.Scorer = {
            'in_dim': self.gene_dim,
            'hidden_dim': [512, 256]
        }


class AdaptConfigs:
    def __init__(self, gene_dim):
        self.gene_dim = gene_dim

        # Training
        self.n_epochs = 50
        self.batch_size = 128
        self.learning_rate = 1e-4
        self.n_critic = 3
        self.loss_weight = {'w_rec': 30, 'w_adv': 1, 'w_gp': 10}
        self.device = select_device('cuda')
        self.random_state = 2024

        # model
        self.Discriminator = {
            'in_dim': self.gene_dim,
            'hidden_dim': [256, 16],
            'num_blocks': 1
        }

        self.Generator = {
            'in_dim': self.gene_dim,
            'hidden_dim': [512, 256],
            'num_blocks': 2,   
        }


class SubtypeConfigs:
    def __init__(self, gene_dim):
        self.gene_dim = gene_dim

        self.n_epochs = 100
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.device = select_device('cuda')
        self.random_state = 2024

        # model
        self.Cluster = {
            'alpha': 1,
            'KMeans_n_init': 20,
            'num_layers': 3,
            'nheads': 4,
            'hidden_dim': 512,
            'dropout': 0.1
        }