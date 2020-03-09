import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from models.tgcn import ConvTemporalGraphical
from models.graph import Graph

from .stgcn import STGCN as ST_GCN
from .group_gcn import Group_GCN as GCN
from models.gat import GAT

class STGCN_2STREAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.join_stream = ST_GCN(cfg)
        self.group_stream = ST_GCN(cfg)
        self.gcn = GCN(cfg)

    def forward(self, x):
        batch = x.shape[0]
        groupSize = 3

        jointInput = x[:, :, :, :, 0]
        N, C, T, V = jointInput.shape
        jointInput = jointInput.reshape((N, C, T, V, 1))
        groupInputs = x[:, :, :, :, 1:]

        joinBodyOutput = self.join_stream(jointInput)
        allBodyOutputs = []
        allBodyOutputs.append(joinBodyOutput)

        for i in range(groupSize):
            groupInput = groupInputs[:, :, :, :, i]
            N, C, T, V = groupInput.shape
            groupInput = groupInput.reshape((N, C, T, V, 1))
            groupBodyOutput = self.group_stream(groupInput)
            allBodyOutputs.append(groupBodyOutput)

        allGroupData = torch.stack(allBodyOutputs, dim=2)
        
        # allGroupData = allGroupData.permute(0, 2, 1).contiguous()

        allGroupData = allGroupData.reshape((N, 16, 1, 4, 1))
        out = self.gcn(allGroupData)
    
        return out