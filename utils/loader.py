import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader

from dataset.data import Data
from dataset.congreg8 import CongreG8

def get_dataset(cfg, mode):
    """ Get dataset by cfg file and mode.
    
    Arguments:
        cfg {cfg} -- config object.
        mode {str} -- training state.
    
    Returns:
        dataset -- dataset for particular name and mode.
    """
    if cfg.TRAIN.DATASET == "data":
        dataset = Data(cfg, mode)
    elif cfg.TRAIN.DATASET == "cater":
        dataset = CATER(cfg, mode)
    elif cfg.TRAIN.DATASET == "congreg8":
        dataset = CongreG8(cfg, mode)
    else:
        raise ValueError("No such dataset")
    return dataset


def get_dataloader(cfg, mode):
    """ Get dataloader by config object and mode. 

    Arguments:
        cfg {cfg} -- config file.
        mode {str} -- different training state.
    
    Returns:
        dataloader, size -- return dataloader for the particular mode and its size.
    """
    dataset = get_dataset(cfg, mode)

    if mode in ['val', 'train']:
        batch_size = cfg.TRAIN.BATCH_SIZE// cfg.NUM_GPUS
        shuffle = True
        drop_last = True
    elif mode == 'test':
        batch_size = cfg.TEST.BATCH_SIZE// cfg.NUM_GPUS
        shuffle = False
        drop_last = False
    else:
        raise ValueError("No such mode")
    
    sampler = DistributedSampler(dataset,shuffle=shuffle) if cfg.NUM_GPUS > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        sampler = sampler,
        drop_last=drop_last
    )
    return dataloader, len(dataloader)


def shuffle_loader(loader, cur_epoch, cfg):
    """
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)


def loader_builder(cfg, mode):
    """ Build dataloader by config object and mode.
    
    Arguments:
        cfg {cfg} -- config object.
        mode {str} -- state for training.
    
    Returns:
        dataloader -- particular dataloader.
    """
    return get_dataloader(cfg, mode)
