import warnings
import torch
import random
import numpy as np
import pandas as pd
from sklearn import metrics


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def clear_warnings(func, category=FutureWarning):
    def warp(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=category)
            temp = func(*args, **kwargs)
            return temp
    return warp


def evaluate(y_true, y_score):
    """calculate evaluation metrics"""
    y_true = pd.Series(y_true)
    y_score = pd.Series(y_score)

    roc_auc = metrics.roc_auc_score(y_true, y_score)
    ap = metrics.average_precision_score(y_true, y_score)
    
    ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    thres = np.percentile(y_score, ratio)
    y_pred = (y_score >= thres).astype(int)
    y_true = y_true.astype(int)
    _, _, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return roc_auc, ap, f1