import torch
from torch import nn
import numpy as np
from utils.misc import params_count, log_model_info, load_checkpoint
from models.construct import model_builder
from utils.loader import loader_builder, shuffle_loader
import utils.distributed as dist
import utils.metrics as metrics 
import utils.losses as losses
import utils.optimizer as optim
import utils.logger as _logger 

logger = _logger.get_logger(__name__)


def multi_view_test(test_loader, model, test_meter, cfg):
    
    # Enable eval mode. 
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx) in enumerate(test_loader):
        
        # Transfer the data to the current GPU device.
        inputs = inputs.cuda()
        labels = labels.cuda()
        video_idx = video_idx.cuda()
        
        # Perform the forward pass.
        preds = model(inputs)
        
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = dist.all_gather([preds, labels, video_idx])
        test_meter.iter_toc()
        
        # Update and log stats
        test_meter.update_stats(
            preds.detach().cpu(),
            labels.detach().cpu(),
            video_idx.detach().cpu(),
        )
        test_meter.log_iter_stats(cur_iter)
        test_meter.iter_tic()
        test_meter.finalize_metrics_map()
        test_meter.reset()
    
def test(cfg):

    # set seed
    torch.cuda.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    _logger.setup_logging(cfg)

    # Print cfg
    logger.info(cfg)

    # Build model and print model info
    model = model_builder(cfg).cuda(device=torch.cuda.current_device())
    
    if cfg.NUM_GPUS > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, 
            device_ids=[torch.cuda.current_device()], 
            output_device=torch.cuda.current_device()
        )

    # Load checkpoint
    load_checkpoint(
        cfg.TEST.LOAD_PATH,
        model,
        cfg.NUM_GPUS > 1,
        None,
    )

    # Build loader
    test_loader, test_size = loader_builder(cfg, 'test')

    # Build meters
    test_meter = metrics.TestMeter(
        len(test_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        test_size,
    )

    logger.info("Testing model for {} iterations".format(len(test_loader)))

    multi_view_test(test_loader, model, test_meter, cfg)
