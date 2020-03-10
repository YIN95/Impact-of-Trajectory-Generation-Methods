import torch
from torch import nn
import numpy as np
from utils.misc import params_count, log_model_info, is_checkpoint_epoch, is_eval_epoch, save_checkpoint, load_checkpoint
from models.construct import model_builder
from utils.loader import loader_builder, shuffle_loader
import utils.distributed as dist
import utils.metrics as metrics 
import utils.losses as losses
import utils.optimizer as optim
import utils.logger as _logger 

logger = _logger.get_logger(__name__)


def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, train_size, tflogger):
    
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    
    for cur_iter, (inputs, labels, _) in enumerate(train_loader):
        # print(cur_iter)
        #if train with RL
        inputs = inputs.float().cuda()
        labels = labels.float().cuda()

        if cfg.MODEL.LOSS_FUNC == "cross_entropy":
            labels = labels.long()
            
        # Update the learning rate.
        lr = optim.get_cur_lr(cur_epoch + float(cur_iter) / train_size, cfg)
        optim.set_lr(optimizer, lr)
        
        # Perform the forward pass.
        preds = model(inputs)
        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)()

        # Compute the loss.
        loss = loss_fun(preds, labels)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()
        
        # num_topks_correct = metrics.topks_correct(preds, labels, (1,))
        # top1_acc = (num_topks_correct[0] / preds.size(0)) * 100.0
    
        if cfg.NUM_GPUS > 1:
            loss = dist.all_reduce([loss])
        print(loss)
        loss = loss[0].item()

        tflogger.add_scalar("loss",loss, train_size*cur_epoch+cur_iter)
        # tflogger.add_scalar("train_top1_acc",top1_acc, train_size*cur_epoch+cur_iter)

        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats_topk(
            loss, lr, inputs.size(0) * cfg.NUM_GPUS
        )
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, val_size):
    # Enable train mode.
    model.eval()
    val_meter.iter_tic()
    
    for cur_iter, (inputs, labels, _) in enumerate(val_loader):
        
        # if eval with RL
        inputs = inputs.float().cuda()
        labels = labels.cuda()

        # Perform the forward pass.
        preds = model(inputs)
        
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)()

        # Compute the loss.
        loss = loss_fun(preds, labels)
        
        # Compute the mAP of current mb.
        # num_topks_correct = metrics.topks_correct(preds, labels, (1,))
        # top1_acc = (num_topks_correct[0] / preds.size(0)) * 100.0
    
        if cfg.NUM_GPUS > 1:
            loss = dist.all_reduce([loss])
        
        loss = loss.item()
        
        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats_topk(
            loss, inputs.size(0) * cfg.NUM_GPUS
        )
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()
    
    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def train(cfg):
    
    # set seed
    torch.cuda.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    _logger.setup_logging(cfg)
    
    # Print cfg
    logger.info(cfg)

    # tensorboard writter
    tflogger = _logger.TFLogger(cfg)

    # Build model and print model info
    model = model_builder(cfg).cuda(device=torch.cuda.current_device())

    if cfg.NUM_GPUS > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, 
            device_ids=[torch.cuda.current_device()], 
            output_device=torch.cuda.current_device()
        )
    
    if dist.is_master_proc(cfg.NUM_GPUS):
        log_model_info(model)
        
    # Build optimizer
    optimizer = optim.optimizer_builder(model, cfg)
    
    # Resume
    start_epoch = 0
    if cfg.TRAIN.RESUME:
        if cfg.TRAIN.LOAD_PATH != "":
            checkpoint_epoch = load_checkpoint(
                cfg.TRAIN.LOAD_PATH,
                model,
                cfg.NUM_GPUS > 1,
                optimizer
            )
            start_epoch = checkpoint_epoch + 1
    
    # Build loader
    train_loader, train_size = loader_builder(cfg, 'train') 
    val_loader  , val_size   = loader_builder(cfg, 'val' )

    # Build meters
    train_meter = metrics.TrainMeter(train_size, cfg)
    val_meter = metrics.TrainMeter(val_size, cfg)

    # Start training
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        
        # Shuffle the data loader
        shuffle_loader(train_loader, cur_epoch, cfg)
        
        # Train the model
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, train_size, tflogger)

        # Save checkpoint
        if is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            save_checkpoint(cfg.CHECKPOINT_DIR, model, optimizer, cur_epoch, cfg)
        
        # Evaluate the model on validation set.
        if is_eval_epoch(cur_epoch, cfg.TRAIN.EVAL_PERIOD, cfg.SOLVER.MAX_EPOCH):
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, val_size)
    
    tflogger.close()