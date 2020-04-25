from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision
import os.path as osp
import os
import numpy as np
import random
import torch
from glob import glob
from . import utils
import pickle as pkl
import utils.logger as _logger

logger = _logger.get_logger(__name__)

class CongreG8(Dataset):
    def __init__(self, cfg, mode='train'):
        self.M = 1
        self.N = 1
        # Only support train, val, and test mode.
        assert mode in ["train", "val"], \
            "Split '{}' not supported for CongreG8".format(mode)

        self.mode = mode
        self.cfg = cfg
        self.ext = ["pkl"]
        
        logger.info("Load CongreG8 {}...".format(mode))
        self._load_data()
        logger.info("Successfully load CongreG8 {} (size:{}).".format(
            self.mode, len(self.label)))
    
    def _load_data(self):
        
        data_path = osp.join(
            self.cfg.DATA_PATH,
            self.mode+'_data.pkl'
        )
        
        label_path = osp.join(
            self.cfg.DATA_PATH,
            self.mode+'_label.pkl'
        )
        
        # load
        file = open(data_path, 'rb')
        self.data = pkl.load(file)
        file.close()

        file = open(label_path, 'rb')
        self.label = pkl.load(file)
        file.close()
        
        # self.N, self.C, self.T, self.V, self.M = self.data.shape
        # self.M = 1

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
            
        data_input = np.array(self.data[index][0])
        data_output = np.array(self.label[index])
        data_output = data_output[:, 0, :]
        # data_output_v = data_output.copy()

        # for i in range(data_output.shape[1]-1):
        #     data_output_v[:, i+1] =  data_output[:,i+1]-data_output[:,i]

        # data_output = data_output.swapaxes(0,1)
        self.C, self.T, self.V, _ = data_input.shape
        
        return data_input, data_output, index
