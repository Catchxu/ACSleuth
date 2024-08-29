import os
import sys
sys.path.insert(0, (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))

import numpy as np
import pandas as pd

from sleuth import load_pkl, AnomalyModel
from sleuth.configs import AnomalyConfigs
from sleuth._utils import update_configs_with_args, evaluate


if __name__ == '__main__':
    data_path = './data/TME_3000.pkl'
    ref, tgt, label = load_pkl(data_path)

    configs = AnomalyConfigs(gene_dim=3000)
    w_recs = [1, 10, 20, 30, 40, 50]
    w_adv = 1
    w_gps = [1, 10, 20, 30, 40, 50]

    result_data = []
    for w_rec in w_recs:
        for w_gp in w_gps:
            new_loss ={'loss_weight': {'w_rec': w_rec, 'w_adv': w_adv, 'w_gp': w_gp}}
            update_configs_with_args(configs, new_loss)

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
            result_data.append(config_dict)
    
            results_df = pd.DataFrame(result_data, index=np.arange(len(result_data)))
            results_df.to_csv('./result/Sensitive.csv', index=False)