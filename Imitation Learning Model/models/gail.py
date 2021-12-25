import torch
from torch import nn

class GAILPolicy(nn.Module):
    def __init__(self, cfg):
        super(GAILPolicy, self).__init__()

        se