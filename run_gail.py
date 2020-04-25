import torch
from torch import nn
import math

import numpy as np
from utils.misc import params_count, log_model_info, is_checkpoint_epoch, is_eval_epoch, save_checkpoint, load_checkpoint
from models.construct import model_builder
from utils.loader import loader_builder, shuffle_loader
import utils.distributed as dist
import utils.metrics as metrics 
import utils.losses as losses
import utils.optimizer as optim
import utils.logger as _logger 
from utils.ppo import ppo_step
from models.value import Value
from models.discriminator import Discriminator
from models.value import Value
from models.policy import Policy_net
from utils.ppo import ppo_step
from utils.estimate_advantages import estimate_advantages
logger = _logger.get_logger(__name__)

def main_gail_loop(cfg):
    # set seed
    torch.cuda.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    _logger.setup_logging(cfg)
    
    # Print cfg
    logger.info(cfg)

    # tensorboard writter
    tflogger = _logger.TFLogger(cfg)

    # Build model and print model info
    Policy = Policy_net(cfg).cuda(device=torch.cuda.current_device())
    # Old_Policy = Policy_net(cfg).cuda(device=torch.cuda.current_device())
    # PPO = PPOTrain(Policy, Old_Policy)
    D = Discriminator(34).cuda(device=torch.cuda.current_device()) 
    V = Value(32).cuda(device=torch.cuda.current_device()) 
    
    optimizer_policy = optim.optimizer_builder(Policy, cfg)
    optimizer_value = optim.optimizer_builder(V, cfg)
    optimizer_discrim = optim.optimizer_builder(D, cfg)
    discrim_criterion = nn.BCELoss()

    train_loader, train_size = loader_builder(cfg, 'train') 
    val_loader, val_size = loader_builder(cfg, 'val') 
    
    if dist.is_master_proc(cfg.NUM_GPUS):
        log_model_info(Policy)
    
    start_epoch = 0

    # Build meters
    train_meter = metrics.TrainMeter(train_size, cfg)
    val_meter = metrics.TrainMeter(val_size, cfg)
    
    # Start training
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the data loader
        shuffle_loader(train_loader, cur_epoch, cfg)
                
        for cur_iter, (inputs, labels, _) in enumerate(train_loader):
            # print(cur_iter)
            logger.info("cur_iter: {}".format(cur_iter + 1))
            rewards = []

            inputs = inputs.float().cuda()
            labels = labels.float().cuda()
    
            states, actions = Policy(inputs, labels[0, :, 0])
            states = torch.stack(states).view(-1,32)
            
            # values
            with torch.no_grad():
                values = V(states)
                values = values[0:-1]
            
            # v_preds = v[0:-1]
            # v_preds_next = v[1:]
            
            # states
            states = states[0:-1]
            # actions
            actions = torch.stack(actions).view(-1,2)[0:-1]
            expert_actions = labels[0, :, 1:-1].permute(1,0)
            
            states_actions = torch.cat([states, actions], dim=1)
            # with torch.no_grad():
            rews = D(states_actions)
            for i in range(len(rews)):
                rewards.append(-math.log(rews[i].item()))
            
            # rewards
            rewards = torch.FloatTensor(rewards).view(-1,1).cuda()
            
            # update discriminator
            for _ in range(1):
                g_o = D(states_actions)
                e_o = D(states_actions)
                optimizer_discrim.zero_grad()
                loss_1 = discrim_criterion(g_o, torch.ones((states.shape[0], 1), device=torch.cuda.current_device()))
                loss_2 = discrim_criterion(e_o, torch.zeros((states.shape[0], 1), device=torch.cuda.current_device()))
                discrim_loss = loss_1 + loss_2
                discrim_loss.backward(retain_graph=True)
                optimizer_discrim.step()
                pass
            
            fixed_log_probs = Policy.get_log_prob(actions)
            
            advantages, returns = estimate_advantages(rewards, values, 0.99, 0.95)
            
            
            # update ppo
            pass
            optim_epochs = 1
            optim_batch_size = 256
            optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
            
            for _ in range(optim_epochs):
                # perm = np.arange(states.shape[0])
                # perm = torch.LongTensor(perm).cuda()
                for i in range(optim_iter_num):
                    ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
                    states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                        states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]
                    
                    ppo_step(Policy, V, optimizer_policy, optimizer_value, states_b, actions_b, returns_b, advantages_b, fixed_log_probs_b, 0.2, 1e-4)
            
            
        # Save checkpoint
        # if is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            save_checkpoint(cfg.CHECKPOINT_DIR, Policy, optimizer_policy, cur_epoch, cfg)
    
    tflogger.close()
