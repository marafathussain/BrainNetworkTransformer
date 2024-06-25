from datetime import datetime
import wandb
import hydra
import random
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from .dataset import dataset_factory
from .models import model_factory
from .components import lr_scheduler_factory, optimizers_factory, logger_factory
from .training import training_factory
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

def model_training(cfg: DictConfig):
    
    scores = ['fiq', 'piq', 'viq', 'fiq_r', 'piq_r', 'viq_r']
    
    for score in scores:
        with open_dict(cfg):
            cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")
            cfg.score = score
            
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        dataloaders = dataset_factory(cfg)
            
        kf5 = KFold(n_splits = 5, random_state = 1, shuffle = True)
        data_index = [i for i in range(len(dataloaders.dataset))]
            
        f = 0
        for fold in kf5.split(data_index):
            training_index = fold[0]
            validation_index = fold[1]
                
            f += 1
                
            # Create Subset objects for training and validation sets
            train_subset = Subset(dataloaders.dataset, training_index)
            validation_subset = Subset(dataloaders.dataset, validation_index)

            # Create new dataloaders for training and validation
            train_dataloader = DataLoader(train_subset, batch_size=cfg.dataset.batch_size, shuffle=True)
            validation_dataloader = DataLoader(validation_subset, batch_size=cfg.dataset.batch_size, shuffle=False)
            
            dataloader = [train_dataloader, validation_dataloader]

            logger = logger_factory(cfg)
            model = model_factory(cfg)
            optimizers = optimizers_factory(model=model, optimizer_configs=cfg.optimizer)
            lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
            training = training_factory(cfg, model, optimizers, lr_schedulers, dataloader, logger, f)

            training.train()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    group_name = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.datasz.percentage}_{cfg.preprocess.name}"
    # _{cfg.training.name}\
    # _{cfg.optimizer[0].lr_scheduler.mode}"

    for _ in range(cfg.repeat_time):
        # run = wandb.init(project=cfg.project, entity=cfg.wandb_entity, reinit=True, group=f"{group_name}", tags=[f"{cfg.dataset.name}"])
        run = wandb.init(project=cfg.project, reinit=True, group=f"{group_name}", tags=[f"{cfg.dataset.name}"])
        #print(cfg)
        model_training(cfg)

        run.finish()


if __name__ == '__main__':
    main()
