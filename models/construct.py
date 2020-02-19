from models.slowfast import SlowFast

def model_builder(cfg):
    if cfg.MODEL.ARCH == 'slowfast':
        return SlowFast(cfg)
    else:
        raise ValueError("No such model!")