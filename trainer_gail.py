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
from models.value import Value
from models.discriminator import Discriminator

logger = _logger.get_logger(__name__)


def expert_reward(state, action):
    state_action = tensor(np.hstack([state, action]), dtype=dtype)
    with torch.no_grad():
        return -math.log(discrim_net(state_action)[0].item())
    
def train_epoch(train_loader, policy_net, value_net, 
                optimizer_policy, optimizer_value, optimizer_discrim, 
                train_meter, cur_epoch, cfg, train_size, tflogger):
    
    # Enable train mode.
    policy_net.train()
    value_net.train()
    train_meter.iter_tic()
    
    for cur_iter, (inputs, labels, _) in enumerate(train_loader):
        # print(cur_iter)
        #if train with RL
        inputs = inputs.float().cuda()
        labels = labels.float().cuda()
        
        # Update the learning rate.
        lr = optim.get_cur_lr(cur_epoch + float(cur_iter) / train_size, cfg)
        optim.set_lr(optimizer_policy, lr)
        optim.set_lr(optimizer_value, lr)
        
        # Perform the forward pass.
        states, actions, rewards, masks = policy_net(inputs, labels[0, :, 0])

        states = torch.cat(states, axis=0)
        actions = torch.cat(actions, axis=0)
        rewards = torch.cat(rewards, axis=0)
        masks = torch.from_numpy(np.stack(masks).reshape(-1, 1)).cuda()


        with torch.no_grad():
            values = value_net(states)
        
        truth = labels.permute(0, 2, 1).contiguous()
        action = action.view(1, len(action), 2)
        

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
    
    
    
def main_loop(cfg):
    # set seed
    torch.cuda.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    _logger.setup_logging(cfg)
    
    # Print cfg
    logger.info(cfg)

    # tensorboard writter
    tflogger = _logger.TFLogger(cfg)

    # Build model and print model info
    policy_net = model_builder(cfg).cuda(device=torch.cuda.current_device())
    value_net = Value(32).cuda(device=torch.cuda.current_device())
    # discrim_net = Discriminator(128).cuda(device=torch.cuda.current_device())

    if cfg.NUM_GPUS > 1:
        policy_net = torch.nn.parallel.DistributedDataParallel(
            module=policy_net, 
            device_ids=[torch.cuda.current_device()], 
            output_device=torch.cuda.current_device()
        )
    
    if dist.is_master_proc(cfg.NUM_GPUS):
        log_model_info(policy_net)
        
    # Build optimizer
    optimizer_policy = optim.optimizer_builder(policy_net, cfg)
    optimizer_value = optim.optimizer_builder(policy_net, cfg)
    optimizer_discrim = optim.optimizer_builder(policy_net, cfg)
    
    start_epoch = 0
    
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
        train_epoch(train_loader, policy_net, value_net, 
                    optimizer_policy, optimizer_value, optimizer_discrim, 
                    train_meter, cur_epoch, cfg, train_size, tflogger)

        # Save checkpoint
        if is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            save_checkpoint(cfg.CHECKPOINT_DIR, policy_net, optimizer, cur_epoch, cfg)
        
        # # Evaluate the model on validation set.
        # if is_eval_epoch(cur_epoch, cfg.TRAIN.EVAL_PERIOD, cfg.SOLVER.MAX_EPOCH):
        #     eval_epoch(val_loader, policy_net, val_meter, cur_epoch, cfg, val_size)
    
    tflogger.close()


