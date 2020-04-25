import torch
import numpy as np
import torch.nn.functional as F
import math
from torch import nn
from .stgcn import STGCN as ST_GCN
from .group_gcn import Group_GCN as GCN
from torch.autograd import Variable
from .value import Value
from .discriminator import Discriminator

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        
class Policy_net(nn.Module):
    def __init__(self, cfg):
        super(Policy_net, self).__init__()
        self.groupSize = 3
        self.group_stream = ST_GCN(cfg)
        self.lstm = nn.LSTM(input_size=2, hidden_size=8, num_layers=2)
        self.traj = []
        self.fc1 = nn.Linear(cfg.STGCN.OUT_FEATURES*3+8, 8)
        self.fc2 = nn.Linear(8, 2)

    def get_log_prob(self, actions):
        action_mean = actions.mean(dim=0).view(1, 2)
        action_log_std = nn.Parameter(torch.ones(1, 2) * 0).cuda()
        action_log_std = action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std).cuda()
        var = action_std.pow(2).cuda()
        log_density = -(actions - action_mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - action_log_std
        return log_density.sum(1, keepdim=True)

    def forward(self, x, x0):
        # N C T V 3
        T = x.shape[2]
        groupInputs = x 
        
        self.traj = []
        self.traj.append(x0.view(1, 2))
        
        self.actions = []
        self.states = []
        self.values = []
        
        for t in range(T-1):
            allBodyOutputs = []
            self.action = torch.cat(self.traj, dim=0)

            for i in range(self.groupSize):
                groupInput = groupInputs[:, :, t, :, i]
                N, C, V = groupInput.shape
                groupInput = groupInput.reshape((N, C, 1, V, 1))
                groupBodyOutput = self.group_stream(groupInput)
                allBodyOutputs.append(groupBodyOutput)
                
            # p1 = Point(groupInputs[0, 0, t, 5, 0], groupInputs[0, 2, t, 5, 0])
            # p2 = Point(groupInputs[0, 0, t, 5, 1], groupInputs[0, 2, t, 5, 1])
            # p3 = Point(groupInputs[0, 0, t, 5, 2], groupInputs[0, 2, t, 5, 2])
            # r_x = (p1.x + p2.x + p3.x) / 3.0
            # r_y = (p1.y + p2.y + p3.y) / 3.0

            # d_join = ((join_next_x[0][0] - r_x) ** 2 + (join_next_x[0][1] - r_y) ** 2)
            # d_1 = ((p1.x - r_x) ** 2 + (p1.y - r_y) ** 2)
            # d_2 = ((p2.x - r_x) ** 2 + (p2.y - r_y) ** 2)
            # d_3 = ((p3.x - r_x) ** 2 + (p3.y - r_y) ** 2)
            # d_max = max(d_1, d_2, d_3)
            
            join_out, (h_n, h_c) = self.lstm(self.action.view(1, len(self.traj), 2), None)
            join_out = join_out[:, -1, :]
            group_out = torch.cat(allBodyOutputs, dim=1)
            state = torch.cat([join_out, group_out], dim=1)
            
            self.states.append(state)

            state_fc1 = self.fc1(state)
            join_next_x = self.fc2(state_fc1)
            
            self.traj.append(join_next_x.view(1,2))
            self.actions.append(join_next_x.view(1,2))
        
        return self.states, self.actions
        