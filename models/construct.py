from models.slowfast import SlowFast
from models.stgcn_twostream import STGCN_2STREAM
from models.gail import GAILPolicy

def model_builder(cfg):
    if cfg.MODEL.ARCH == 'slowfast':
        return SlowFast(cfg)
    elif cfg.MODEL.ARCH == 'stgcn':
        return STGCN_2STREAM(cfg)
    elif cfg.MODEL.ARCH == 'gail':
        return GAILPolicy(cfg)
    else:
        raise ValueError("No such model!")