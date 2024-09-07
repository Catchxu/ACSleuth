import os
import sys
sys.path.insert(0, (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))

import numpy as np
import pandas as pd

from sleuth import load_pkl, AnomalyModel
from sleuth.configs import AnomalyConfigs
from sleuth._utils import update_configs_with_args, evaluate


if __name__ == '__main__':
    data_path = './data/Cancer_3000.pkl'
    ref, tgt, label = load_pkl(data_path)

    configs = AnomalyConfigs(gene_dim=3000)
    mem_dim = [256, 384, 512, 640, 768, 896]

    result_data = []
    for m in mem_dim:
        configs.Generator['mem_dim'] = m

        model = AnomalyModel(configs)
        model.detect(ref)
        score = model.predict(tgt)

        roc_auc, ap, f1 = evaluate(label, score)
        config_dict = {
            'mem_dim': m,
            'roc_auc': roc_auc,
            'ap': ap,
            'f1': f1
            }
        result_data.append(config_dict)

        results_df = pd.DataFrame(result_data, index=np.arange(len(result_data)))
        results_df.to_csv('./result/MemoryBank.csv', index=False)