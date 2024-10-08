import os
import sys
sys.path.insert(0, (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))

import argparse
import pandas as pd
from typing import Dict

from sleuth import load_pkl, AnomalyModel
from sleuth.configs import AnomalyConfigs
from sleuth._utils import update_configs_with_args, evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ACSleuth for anomaly detection.')

    parser.add_argument('--data_name', type=str, default='TME_3000', help='Name of the saved dataset to be loaded.')
    parser.add_argument('--gene_dim', type=int, default=3000, help='Dimension or number of genes.')
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
    name = args_dict['data_name']
    ref, tgt, label = load_pkl(f'./data/{name}.pkl')

    # update configs
    configs = AnomalyConfigs(args_dict['gene_dim'])
    update_configs_with_args(configs, args_dict)

    model = AnomalyModel(configs)
    model.detect(ref)
    score = model.predict(tgt)

    roc_auc, ap, f1 = evaluate(label, score)

    config_dict = configs.to_dict()
    config_dict.update({
        'roc_auc': roc_auc,
        'ap': ap,
        'f1': f1
    })
    results_df = pd.DataFrame(config_dict, index=[0])
    results_df.to_csv(f'./result/{name}.csv', index=False)