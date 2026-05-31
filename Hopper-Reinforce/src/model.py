import torch
import torch.nn as nn
from torch.distributions import Normal

# Since the torque is a continuous variable, we use our network for a regression task, having the output layer predicting the [-1,1] value of each joint's torque.
# However, since we want our policy to explore as well instead of greedily doing what the MLP output suggests, we add the St.Deviation as a learnable parameter! In this way we have our agent to not focus always on the output of the MLP, using the MLP output as a mean, the learnable param as std deviation, and sampling an action from a Gaussian distribution with those two params.

class Policy(nn.Module):
  def __init__(self,obs_space_size,act_space_size) -> None:
    super().__init__()

    self.obs_space_size=obs_space_size
    self.act_space_size=act_space_size

    self.MLP=nn.Sequential(
        nn.Linear(self.obs_space_size,32),
        nn.ReLU(),
        nn.Linear(32,64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,self.act_space_size),
        nn.Tanh()
    )

    self.std=nn.Parameter(torch.zeros(1,act_space_size))

  def forward(self,x):
    mean=self.MLP(x)
    std=self.std.exp().expand_as(mean) #add exp to keep std positive, expand_as is needed to handle batch size
    return mean,std

  def act(self,state,device):
    state=torch.from_numpy(state).float().unsqueeze(0).to(device)
    mean,std=self.forward(state)

    m=Normal(mean,std)
    action=m.sample()

    action_np=action.detach().cpu().numpy()[0] #convert from torch tensor to numpy array, detach from computational graph
    log_prob=m.log_prob(action).sum(dim=-1) #score function
    entropy=m.entropy().sum(dim=-1)

    return action_np,log_prob,entropy
