from models.slowfast import SlowFast
from models.stgcn_twostream import STGCN_2STREAM

def model_builder(cfg):
    if cfg.MODEL.ARCH == 'slowfast':
        return SlowFast(cfg)
    elif cfg.MODEL.ARCH == 'stgcn':
        return STGCN_2STREAM(cfg)
    else:
        raise ValueError("No such model!")