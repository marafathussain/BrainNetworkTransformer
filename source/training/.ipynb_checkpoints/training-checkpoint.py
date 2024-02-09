from source.utils import accuracy, TotalMeter, count_params, isfloat, mse, mae
import torch
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
from torch.utils.data import DataLoader


class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        #self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.loss_fn = torch.nn.MSELoss()
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph

        self.init_meters()

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

            label = label.unsqueeze(1) # I added
            loss = self.loss_fn(predict, label)

            #self.train_loss.update_with_weight(loss.item(), label.shape[0])
            self.train_loss.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #top1 = accuracy(predict, label[:, 1])[0]
            #self.train_accuracy.update_with_weight(top1, label.shape[0])
            mae_error = mae(predict, label)
            #self.train_accuracy.update_with_weight(mse_error, label.shape[0])
            self.train_accuracy.update(mae_error)
            # wandb.log({"LR": lr_scheduler.lr,
            #            "Iter loss": loss.item()})

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for time_series, node_feature, label in dataloader:
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            output = self.model(time_series, node_feature)

            label = label.float()
            
            label = label.unsqueeze(1)
            loss = self.loss_fn(output, label)
            #loss_meter.update_with_weight(loss.item(), label.shape[0])
            loss_meter.update(loss.item())
            #top1 = accuracy(output, label[:, 1])[0]
            #acc_meter.update_with_weight(top1, label.shape[0])
            mae_error = mae(output, label)
            #acc_meter.update_with_weight(mse_error, label.shape[0])
            acc_meter.update(mae_error)
        '''
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()

        auc = roc_auc_score(labels, result)
        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')

        report = classification_report(
            labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall
        '''
        return mae_error

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

    def train(self):
        training_process = []
        self.current_step = 0
        for epoch in range(self.epochs):
            self.reset_meters()
            
            # =============== Vanilla BNT ====================
            #self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            #val_result = self.test_per_epoch(self.val_dataloader, self.val_loss, self.val_accuracy)
            #test_result = self.test_per_epoch(self.test_dataloader, self.test_loss, self.test_accuracy)
            
            # =============== Relational BNT ====================
            self.train_pairs_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            val_result = self.test_pairs_per_epoch(self.val_dataloader, self.val_loss, self.val_accuracy)
            test_result = self.test_pairs_per_epoch(self.test_dataloader, self.test_loss, self.test_accuracy)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}',
                f'val Loss:{self.val_loss.avg: .3f}',
                f'val Accuracy:{self.val_accuracy.avg: .3f}',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}' #,
                #f'Test AUC:{test_result[0]:.4f}',
                #f'Test Sen:{test_result[-1]:.4f}',
                #f'LR:{self.lr_schedulers[0].lr:.4f}'
            ]))

            '''
            wandb.log({
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Val AUC": val_result[0],
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
            })
            '''

            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                #"Test AUC": test_result[0],
                #'Test Sensitivity': test_result[-1],
                #'Test Specificity': test_result[-2],
                #'micro F1': test_result[-4],
                #'micro recall': test_result[-5],
                #'micro precision': test_result[-6],
                #"Val AUC": val_result[0],
                "Val Loss": self.val_loss.avg,
            })
    
    # I added
    #==================================================================================================
    def train_pairs_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()
        
        paired_data = []

        # Get the length of the dataset
        dataset_size = len(self.train_dataloader.dataset)

        # Create all possible permutations for indices
        all_permutations = list(permutations(range(dataset_size), 2))
        
        # Iterate through the permutations to create pairs
        for perm in all_permutations:
            idx1, idx2 = perm

            time_series1, node_feature1, label1 = self.train_dataloader.dataset[idx1]
            time_series2, node_feature2, label2 = self.train_dataloader.dataset[idx2]

            # Combine data points into pairs
            paired_data.append(((time_series1, node_feature1, label1), (time_series2, node_feature2, label2)))

        # Create a new DataLoader for the paired data
        batch_size = 16  # Adjust as needed
        paired_dataloader = DataLoader(paired_data, batch_size=batch_size, shuffle=True)

        # Now you can iterate through paired_dataloader to get pairs of data
        change_config = False
        for pair in paired_dataloader:
            (time_series1, node_feature1, label1), (time_series2, node_feature2, label2) = pair
            label1, label2 = label1.float(), label2.float()
            self.current_step += 1
            
            # Four relational targets
            r1 = (label1 + label2).unsqueeze(1)
            r2 = torch.abs(label1 - label2).unsqueeze(1)
            r3 = torch.max(label1, label2).unsqueeze(1)
            r4 = torch.min(label1, label2).unsqueeze(1)
            
            # Concatenate r1, r2, r3, r4 into combined_label
            combined_label = torch.cat((r1, r2, r3, r4), dim=1)
            
            # Assuming time_series1 and time_series2 have shape (batch_size, seq_len, feature_dim)
            #concatenated_time_series = torch.cat((time_series1, time_series2), dim=1)
            #concatenated_node_feature = torch.cat((node_feature1, node_feature2), dim=1)
            
            # Using .size()
            #print("Size - Time Series:", concatenated_time_series.size())
            #print("Size - Node Features:", concatenated_node_feature.size())

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)
            node_feature1, node_feature2, combined_label = node_feature1.cuda(), node_feature2.cuda(), combined_label.cuda()

            #concatenated_time_series, concatenated_node_feature, combined_label = concatenated_time_series.cuda(), concatenated_node_feature.cuda(), combined_label.cuda()

            predict = self.model(node_feature1, node_feature2)

            #combined_label = combined_label.unsqueeze(1)
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
            
    def test_pairs_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for time_series1, node_feature1, label1 in dataloader:
            for time_series2, node_feature2, label2 in dataloader:
                
                if torch.equal(time_series1, time_series2):
                    continue
                
                label1, label2 = label1.float(), label2.float()
                
                # Four relational targets
                r1 = (label1 + label2).unsqueeze(1)
                r2 = torch.abs(label1 - label2).unsqueeze(1)
                r3 = torch.max(label1, label2).unsqueeze(1)
                r4 = torch.min(label1, label2).unsqueeze(1)
            
                # Concatenate r1, r2, r3, r4 into combined_label
                combined_label = torch.cat((r1, r2, r3, r4), dim=1)
            
                # Assuming time_series1 and time_series2 have shape (batch_size, seq_len, feature_dim)
                concatenated_time_series = torch.cat((time_series1, time_series2), dim=1)
                concatenated_node_feature = torch.cat((node_feature1, node_feature2), dim=1)
                
                concatenated_time_series, concatenated_node_feature, combined_label = concatenated_time_series.cuda(), concatenated_node_feature.cuda(), combined_label.cuda()
                
                output = self.model(concatenated_time_series, concatenated_node_feature)
            
                combined_label = combined_label.unsqueeze(1)
                loss = self.loss_fn(output, combined_label)
                
                loss_meter.update(loss.item())
                output_r1 = output[:, :1]
                mae_error = mae(output_r1, r1)
                acc_meter.update(mae_error/2.0)

        return mae_error/2.0
         #==================================================================================================
         
        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)
