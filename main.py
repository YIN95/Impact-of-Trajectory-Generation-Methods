import argparse
import torch
import utils.multiprocess as mpu
from configs.config import get_config
from trainer import train
from tester import test
from trainer_gail import main_loop
from run_gail import main_gail_loop

def main():
    """
    Main function to spawn the train and test process.
    """
    parser = argparse.ArgumentParser(
        description="Provide training and testing pipeline."
    )
    
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default='configs/congreg8/config.yaml',
        type=str,
    )
    
    parser.add_argument(
        "--ip",
        dest="ip",
        help="The IP of localhost",
        default="tcp://localhost:9999",
        type=str,
    )
    
    parser.add_argument(
        "opts",
        help="See the default yaml file",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    args = parser.parse_args()

    cfg = get_config(args.cfg_file, args.opts)

    test(cfg)
    # Perform training
    if cfg.TRAIN.ENABLE:
        if cfg.NUM_GPUS > 1:
            torch.multiprocessing.spawn(
                mpu.run,
                nprocs=cfg.NUM_GPUS,
                args=(
                    cfg.NUM_GPUS,
                    main_loop,
                    args.ip,
                    cfg.SHARD_ID,       # default: 0 
                    cfg.NUM_SHARDS,     # default: 1
                    cfg.DIST_BACKEND,   # default: nccl
                    cfg,
                ),
                daemon=False,
            )
        else:
            main_gail_loop(cfg)
            
if __name__ == "__main__":
    
    main()
    