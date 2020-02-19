from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision
import os.path as osp
import os
import numpy as np
import random
import torch
from glob import glob
import utils.logger as _logger

logger = _logger.get_logger(__name__)


class Data(Dataset):
    def __init__(self, cfg, mode='train'):
        
        # Only support train, val, and test mode.
        assert mode in ["train", "val", "test", ], \
            "Split '{}' not supported for Data".format(mode)
        
        self.mode = mode
        self.cfg = cfg
        
        logger.info("Load Data {}...".format(mode))
        self._load_data()
        logger.info("Successfully load Data {} (size:{}).".format(
            self.mode, 999))

    def _load_data(self):
        ...
    
    def __len__(self):
        return 999
    
    def __getitem__(self, index):

        if self.mode in ["train", "val"]:
            ...
        
        elif self.mode in ["test"]:
            ...
        
        