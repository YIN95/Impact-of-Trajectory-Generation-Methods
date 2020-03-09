import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.tgcn import ConvTemporalGraphical
from models.graph import Graph
from models.stgcn_module import st_gcn

class Group_GCN(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        
        in_channels = cfg.STGCN.OUT_FEATURES
        num_class = cfg.MODEL.NUM_CLASSES
        edge_importance_weighting = cfg.STGCN.EDGE_IMPORTANCE
        
        self.graph = Graph(cfg, layout="group")
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = cfg.STGCN.TEMPORAL_KERNEL_SIZE
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, stride=1, residual=False, dropout=0),
            st_gcn(64, 64, kernel_size, stride=1, residual=True, dropout=0),
            st_gcn(64, 64, kernel_size, stride=1, residual=True, dropout=0),
        ))
        
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        
        hidden = 8
        self.fcn = nn.Conv2d(64, hidden, kernel_size=1)
        self.fc1 = nn.Linear(hidden, 2)

    def forward(self, x):
    
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
