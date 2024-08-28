import argparse
from typing import Dict

from sleuth import load_pkl, AnomalyModel
from sleuth.configs import AnomalyConfigs
from sleuth._utils import update_configs_with_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ACSleuth for anomaly detection.')

    parser.add_argument('--data_path', type=str, default='./data/TME_3000.pkl', help='Path to read the saved dataset.')
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
    ref, tgt = load_pkl(args_dict['data_path'])

    # update configs
    configs = AnomalyConfigs(args_dict['gene_dim'])
    update_configs_with_args(configs, args_dict)

    model = AnomalyModel(configs)
    model.detect(ref)
    score = model.predict(tgt)