import numpy as np
import torch
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_abide_data(cfg: DictConfig):

    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    final_timeseires = data["timeseires"]
    final_pearson = data["corr"]
    #labels = data["label"]
    labels = data[cfg.score]
    site = data['site']

    scaler = StandardScaler(mean=np.mean(
        final_timeseires), std=np.std(final_timeseires))

    final_timeseires = scaler.transform(final_timeseires)
    
    #--added---
    #scaler2 = StandardScaler(mean=np.mean(final_pearson), std=np.std(final_pearson))
    #final_pearson = scaler2.transform(final_pearson)
    #----------

    final_timeseires, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, labels)]

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]
        
        #cfg.dataset.node_sz = cfg.dataset.node_sz*2

    return final_timeseires, final_pearson, labels #, site
