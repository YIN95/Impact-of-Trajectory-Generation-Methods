import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from .stgcn import STGCN as ST_GCN
from .group_gcn import Group_GCN as GCN
from torch.autograd import Variable
from .discriminator import Discriminator

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
 
def getCircle(p1, p2, p3):
    x21 = p2.x - p1.x
    y21 = p2.y - p1.y
    x32 = p3.x - p2.x
    y32 = p3.y - p2.y
    # three colinear
    if (x21 * y32 - x32 * y21 == 0):
        return None
    xy21 = p2.x * p2.x - p1.x * p1.x + p2.y * p2.y - p1.y * p1.y
    xy32 = p3.x * p3.x - p2.x * p2.x + p3.y * p3.y - p2.y * p2.y
    y0 = (x32 * xy21 - x21 * xy32) / 2 * (y21 * x32 - y32 * x21)
    x0 = (xy21 - 2 * y0 * y21) / (2.0 * x21)
    R = ((p1.x - x0) ** 2 + (p1.y - y0) ** 2) ** 0.5
    return x0, y0, R


class GAILPolicy(nn.Module):
    def __init__(self, cfg):
        super(GAILPolicy, self).__init__()
        self.groupSize = 3
        self.group_stream = ST_GCN(cfg)
        self.lstm = nn.LSTM(input_size=2, hidden_size=8, num_layers=2)
        self.traj = []
        self.fc1 = nn.Linear(cfg.STGCN.OUT_FEATURES*3+8, 8)
        self.fc2 = nn.Linear(8, 2)
        self.discrim_net = Discriminator(32)
        
    def forward(self, x, x0):
        # N C T V 3
        T = x.shape[2]
        groupInputs = x 
        self.traj = []
        self.traj.append(x0.view(1, 2))
        self.action = []
        self.state = []
        self.rewards = []
        self.masks = []

        for t in range(T-1):
            allBodyOutputs = []
            self.action = torch.cat(self.traj, dim=0)

            for i in range(self.groupSize):
                groupInput = groupInputs[:, :, t, :, i]
                N, C, V = groupInput.shape
                groupInput = groupInput.reshape((N, C, 1, V, 1))
                groupBodyOutput = self.group_stream(groupInput)
                allBodyOutputs.append(groupBodyOutput)
                
            p1 = Point(groupInputs[0, 0, t, 5, 0], groupInputs[0, 2, t, 5, 0])
            p2 = Point(groupInputs[0, 0, t, 5, 1], groupInputs[0, 2, t, 5, 1])
            p3 = Point(groupInputs[0, 0, t, 5, 2], groupInputs[0, 2, t, 5, 2])
            r_x, r_y, r_r = getCircle(p1, p2, p3)

            
            
            join_out, (h_n, h_c) = self.lstm(self.action.view(1, len(self.traj), 2), None)
            join_out = join_out[:, -1, :]
            group_out = torch.cat(allBodyOutputs, dim=1)
            state = torch.cat([join_out, group_out], dim=1)
            self.state.append(state)
            reward = self.discrim_net(state)
            self.rewards.append(reward)
            state_fc1 = self.fc1(state)
            join_next_x = self.fc2(state_fc1)
            self.traj.append(join_next_x.view(1,2))
            
            dis = ((join_next_x[0][0] - r_x) ** 2 + (join_next_x[0][1] - r_y) ** 2) ** 0.5
            if dis < r_r:
                mask = 1
            else:
                mask = 0
                
            self.masks.append(mask)

        # self.action = torch.cat(self.traj, dim=0)
        
        return self.state, self.traj, self.rewards, self.masks
    
    
    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)
