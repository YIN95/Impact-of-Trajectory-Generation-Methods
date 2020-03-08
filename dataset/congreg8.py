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
import utils.logger as _logger

logger = _logger.get_logger(__name__)

class CongreG8(Dataset):
    def __init__(self, cfg, mode='train'):

        # Only support train, val, and test mode.
        assert mode in ["train", "val"], \
            "Split '{}' not supported for CongreG8".format(mode)

        self.mode = mode
        self.cfg = cfg
        self.ext = ["npy"]
        
        logger.info("Load CongreG8 {}...".format(mode))
        self._load_data()
        logger.info("Successfully load CongreG8 {} (size:{}).".format(
            self.mode, len(self.label)))
    
    def _load_data(self):
        # path
        # label_path = osp.join(
        #     self.cfg.DATA_PATH,
        #     'val_label.npy'
        # )
        # data_path = osp.join(
        #     self.cfg.DATA_PATH,
        #     'val_data.npy'
        # )
        label_path = osp.join(
            self.cfg.DATA_PATH,
            self.mode+'_label.npy'
        )
        data_path = osp.join(
            self.cfg.DATA_PATH,
            self.mode+'_data.npy'
        )

        # load
        self.label = np.load(label_path)
        self.data = np.load(data_path, mmap_mode='r+')
        
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        self.M = 1

        assert (
            len(self.label) == self.N
        ), "Failed to load congreg8 split {} from {}".format(
            self.mode, self.cfg.DATA_PATH
        )

    def __len__(self):
        return len(self.label)*(self.cfg.DATA.NUM_FRAMES-1)
    
    def __getitem__(self, index):
        
        data_index = int(index/(self.cfg.DATA.NUM_FRAMES-1))
        frame_index = index - data_index*(self.cfg.DATA.NUM_FRAMES-1)
        
        data_numpy = np.array(self.data[data_index])
        
        data_input = data_numpy[:, frame_index, :, :]
        data_input = data_input[:, np.newaxis, :, :]
    
        data_output = data_numpy[:, frame_index+1, :, :]
        data_output = data_output[:, np.newaxis, :, :]
                
        return data_input, data_output, index
