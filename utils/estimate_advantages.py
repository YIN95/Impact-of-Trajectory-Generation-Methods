  
import torch

def estimate_advantages(rewards, values, gamma, tau):
    # rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    gamma = torch.ones(1).cuda()*gamma
    tau = torch.ones(1).cuda()*tau
    tensor_type = type(rewards)
    
    deltas = tensor_type(rewards.size(0), 1).cuda()
    advantages = tensor_type(rewards.size(0), 1).cuda()

    prev_value = torch.zeros(1).cuda()
    prev_advantage = torch.zeros(1).cuda()

    for i in reversed(range(rewards.shape[0]-1)):
        deltas[i] = rewards[i] + gamma* prev_value * 1 - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * 1

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    deltas[-1] = rewards[-1] + gamma * prev_value * 0 - values[-1]
    advantages[-1] = deltas[-1] + gamma * tau * prev_advantage * 0
        
    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages, returns