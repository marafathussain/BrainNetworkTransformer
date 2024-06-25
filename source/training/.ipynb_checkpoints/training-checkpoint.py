from source.utils import accuracy, TotalMeter, count_params, isfloat, mse, mae
import torch
import random
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data
import wandb
from omegaconf import DictConfig, open_dict
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging
from itertools import permutations
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import copy


class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger,
                 fold: int) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.fold = fold
        #self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders # commented to facilitate a 5-fold CV
        self.train_dataloader, self.val_dataloader = dataloaders # added
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        #self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.loss_fn = torch.nn.MSELoss()
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph
        self.init_meters()
        
        if self.config.model.name == 'RelationalBrainNetworkTransformer' or self.config.model.name == 'RelationalBrainNetworkTransformer2' or self.config.model.name == 'RelationalBrainNetworkTransformer3':
            with open_dict(cfg):
                cfg.training.epochs = 20
            self.epochs = cfg.training.epochs

    def init_meters(self):                     # look for TotalMeter as imported from utils/meters/
        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()

        for time_series, node_feature, label in self.train_dataloader:
            label = label.float()
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()

            if self.config.preprocess.continus:
                time_series, node_feature, label = continus_mixup_data(
                    time_series, node_feature, y=label)

            predict = self.model(time_series, node_feature)

            label = label.unsqueeze(1) 
            loss = self.loss_fn(predict, label)

            self.train_loss.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mae_error = mae(predict, label)
            self.train_accuracy.update(mae_error)


    def test_per_epoch(self, cfg, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for time_series, node_feature, label in dataloader:
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            
            if len(label) != cfg.dataset.batch_size:
                    continue
                
            output = self.model(time_series, node_feature)

            label = label.float()
            
            label = label.unsqueeze(1)
            #print('label shape:', label.shape)
            #print('output shape:', output.shape)
            loss = self.loss_fn(output, label)
            
            labels.append(label.cpu().detach().numpy())
            result.append(output.cpu().detach().numpy())
            
            loss_meter.update(loss.item())
            mae_error = mae(output, label)
            acc_meter.update(mae_error)
        
        #print('labels shape:', labels.shape)
        #print('predicted score:', result.shape)
        return {'label': np.array(labels), 'predicted_score': np.array(result)}

    def generate_save_learnable_matrix(self):

        # wandb.log({'heatmap_with_text': wandb.plots.HeatMap(x_labels, y_labels, matrix_values, show_text=False)})
        learable_matrixs = []

        labels = []

        for time_series, node_feature, label in self.test_dataloader:
            label = label.long()
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            _, learable_matrix, _ = self.model(time_series, node_feature)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)

        torch.save(self.model.state_dict(), self.save_path/"model.pt")


    def train_pairs_per_epoch(self, cfg, train_dataloader, optimizer, lr_scheduler):
        self.model.train()
        
        paired_data = []
        
        # Get the length of the dataset
        dataset_size = len(train_dataloader.dataset)

        # Create all possible permutations for indices
        all_permutations = list(permutations(range(dataset_size), 2))
        
        with open_dict(cfg):
            cfg.steps_per_epoch = ((len(all_permutations) - 1) // cfg.dataset.batch_size) + 1
            cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs
        
        # Iterate through the permutations to create pairs
        for perm in all_permutations:
            idx1, idx2 = perm

            time_series1, node_feature1, label1 = train_dataloader.dataset[idx1]
            time_series2, node_feature2, label2 = train_dataloader.dataset[idx2]

            # Combine data points into pairs
            paired_data.append(((time_series1, node_feature1, label1), (time_series2, node_feature2, label2)))

        # Create a new DataLoader for the paired data
        batch_size = cfg.dataset.batch_size 
        paired_dataloader = DataLoader(paired_data, batch_size=batch_size, shuffle=True)

        # Now you can iterate through paired_dataloader to get pairs of data
        change_config = False
        
        for pair in paired_dataloader:
            (time_series1, node_feature1, label1), (time_series2, node_feature2, label2) = pair
            label1, label2 = label1.float(), label2.float()
            self.current_step += 1
            
            # Four relational targets
            r1 = (label1 + label2).unsqueeze(1)
            r2 = (label1 - label2).unsqueeze(1)
            r3 = torch.max(label1, label2).unsqueeze(1)
            r4 = torch.min(label1, label2).unsqueeze(1)
            
            # Concatenate r1, r2, r3, r4 into combined_label
            combined_label = torch.cat((r1, r2, r3, r4), dim=1)

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)
            node_feature1, node_feature2, combined_label = node_feature1.cuda(), node_feature2.cuda(), combined_label.cuda()
            r1, r2, r3, r4 = r1.cuda(), r2.cuda(), r3.cuda(), r4.cuda()

            predict = self.model(node_feature1, node_feature2)

            loss = self.loss_fn(predict, combined_label)

            self.train_loss.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Extracting r1, r2, r3, r4 from combined_label
            predicted_r1 = predict[:, :1]
            #predicted_r2 = predict[:, 1:2]
            #predicted_r3 = predict[:, 2:3]
            #predicted_r4 = predict[:, 3:4]

            # Extracting predicted label1 and label2 from predicted_r1 and predicted_r2
            #predicted_label1 = (predicted_r2 + predicted_r1) / 2.0
            #predicted_label2 = (predicted_r2 - predicted_r1) / 2.0
            
            mae_error = mae(predicted_r1, r1)
            self.train_accuracy.update(mae_error/2.0)
            
    def test_pairs_per_epoch(self, cfg, validation_dataloader, loss_meter, acc_meter):
        label_1 = []
        label_2 = []
        combined = []
        result = []

        self.model.eval()
        
        dataset_size = len(validation_dataloader.dataset)

        for time_series1, node_feature1, label1 in validation_dataloader:
            for time_series2, node_feature2, label2 in validation_dataloader:
                
                if len(label1) != cfg.dataset.batch_size or len(label2) != cfg.dataset.batch_size:
                    continue
                
                label1, label2 = label1.float(), label2.float()
                
                # Four relational targets
                r1 = (label1 + label2).unsqueeze(1)
                r2 = (label1 - label2).unsqueeze(1)
                r3 = torch.max(label1, label2).unsqueeze(1)
                r4 = torch.min(label1, label2).unsqueeze(1)
                
                # Concatenate r1, r2, r3, r4 into combined_label
                combined_label = torch.cat((r1, r2, r3, r4), dim=1)
            
                node_feature1, node_feature2, combined_label = node_feature1.cuda(), node_feature2.cuda(), combined_label.cuda()
                r1, r2, r3, r4 = r1.cuda(), r2.cuda(), r3.cuda(), r4.cuda()
                
                combined_label = combined_label.squeeze(1)
                output = self.model(node_feature1, node_feature2)
                loss = self.loss_fn(output, combined_label)
                
                label_1.append(label1.cpu().detach().numpy())
                label_2.append(label2.cpu().detach().numpy())
                combined.append(combined_label.cpu().detach().numpy())
                result.append(output.cpu().detach().numpy())
                
                loss_meter.update(loss.item())
                output_r1 = output[:, :1]
                mae_error = mae(output_r1, r1)
                acc_meter.update(mae_error/2.0)

        return {'label1': np.array(label_1), 'label2': np.array(label_2), 'combined_scores': np.array(combined), 'predicted_score': np.array(result)}
        
    def train(self):
        training_process = []
        self.current_step = 0
        
        best_val_loss = np.inf
        #train_losses = []
        #val_losses = []
        val_predictions = []

        for epoch in range(self.epochs):
            self.reset_meters()
            
            if self.config.model.name == 'RelationalBrainNetworkTransformer' or self.config.model.name == 'RelationalBrainNetworkTransformer2' or self.config.model.name == 'RelationalBrainNetworkTransformer3':
                # Relational BNT
                self.train_pairs_per_epoch(self.config, self.train_dataloader, self.optimizers[0], self.lr_schedulers[0])
                val_result = self.test_pairs_per_epoch(self.config, self.val_dataloader, self.val_loss, self.val_accuracy)
            else:
                self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
                val_result = self.test_per_epoch(self.config, self.val_dataloader, self.val_loss, self.val_accuracy)

            if self.val_accuracy.avg < best_val_loss:
                best_val_loss = self.val_accuracy.avg
                val_predictions = val_result

            self.logger.info(" | ".join([
                f'Epoch[{epoch+1}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}',
                f'val Loss:{self.val_loss.avg: .3f}',
                f'val Accuracy:{self.val_accuracy.avg: .3f}'
            ]))

            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Val Loss": self.val_loss.avg,
            })
        
        saving_path = '/home/ch225256/Data/miccai24'
        np.save(f"{saving_path}/{self.config.model.name}_{self.config.score}_k10_CombT_noFC_{self.fold}_validation_predictions.npy", val_predictions)
    
        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)    