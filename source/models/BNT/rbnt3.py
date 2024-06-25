import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from .components import InterpretableTransformerEncoder
from omegaconf import DictConfig
from ..base import BaseModel


class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=4, dim_feedforward=hidden_size, batch_first=True)
        #print('size of d_model:', input_feature_size) --> 200

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size * input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder, orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        #print('size of x before Tansformer:', x.shape) --> [batch_size, 200, 200]
        x = self.transformer(x)
        #print('size of x after Transformer:', x.shape) --> [batch_size, 200, 200]
        if self.pooling:
            x, assignment = self.dec(x)
            #print('size of x after OCRead pooling:', x.shape) --> [batch_size, 100, 200]
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)

class RelationalBrainNetworkTransformer3(BaseModel):

    def __init__(self, config: DictConfig):

        super().__init__()

        self.attention_list1 = nn.ModuleList()  # Attention modules for node_feature1
        self.attention_list2 = nn.ModuleList()  # Attention modules for node_feature2

        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling
        
        for index, size in enumerate(sizes):
            # Attention module for node_feature1
            self.attention_list1.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=in_sizes[index],
                                    hidden_size=1024,
                                    output_node_num=size,
                                    pooling=do_pooling[index],
                                    orthogonal=config.model.orthogonal,
                                    freeze_center=config.model.freeze_center,
                                    project_assignment=config.model.project_assignment))
            
            # Attention module for node_feature2
            self.attention_list2.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=in_sizes[index],
                                    hidden_size=1024,
                                    output_node_num=size,
                                    pooling=do_pooling[index],
                                    orthogonal=config.model.orthogonal,
                                    freeze_center=config.model.freeze_center,
                                    project_assignment=config.model.project_assignment))

        self.combinedDataTransformer = InterpretableTransformerEncoder(d_model=forward_dim, nhead=8, batch_first=True)
        
        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )
        
        self.short_fc = nn.Sequential(
            nn.Linear(160, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1] * 2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self,
            node_feature1: torch.tensor,
            node_feature2: torch.tensor):

        bz, _, _, = node_feature1.shape

        if self.pos_encoding == 'identity':
            pos_emb1 = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature1 = torch.cat([node_feature1, pos_emb1], dim=-1)
            
            pos_emb2 = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature2 = torch.cat([node_feature2, pos_emb2], dim=-1)

        # Process node_feature1 through attention_list1
        for atten1 in self.attention_list1:
            node_feature1, assignment1 = atten1(node_feature1)

        # Process node_feature2 through attention_list2
        for atten2 in self.attention_list2:
            node_feature2, assignment2 = atten2(node_feature2)
        
        # Concatenate the individual outputs along the last dimension
        combined_node_features = torch.cat([node_feature1, node_feature2], dim=1)
        #print('size of node_features before combined transformer:', combined_node_features.shape)  #--> [batch_size, 20, 200]
        
        
        node_feature = self.combinedDataTransformer(combined_node_features)
        #print('size of node_features before dimension reduction:', node_feature.shape) #--> [batch_size, 20, 200]
        
        #node_feature = self.dim_reduction(combined_node_features)
        node_feature = self.dim_reduction(node_feature)
        #print('size of node_features after dimension reduction:', node_feature.shape) #--> [batch_size, 20, 8]

        node_feature = node_feature.reshape((bz, -1))
        #print('size of Z_G:', node_feature.shape) #--> [batch_size, 160]

        #return self.fc(node_feature)
        
        return self.short_fc(node_feature)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all
