import torch
import torch.nn as nn
from torch.distributions import Normal

# The main reason why switching from Reinforce to Actor Critic is high Variance: $G_t$ is the sum of many random events (the actions sampled from your normal distribution and the environment's transitions). A single bad action at step 100 can ruin the total return $G$, unfairly penalizing a great action taken at step 2.
# 
# Instead of waiting for the end of the episode to see how good an action was, we use an "expert" to estimate how good a state is in real-time.
# 
# The Actor is our current Policy network. It decides what to do ($\pi_\theta(a|s)$).The Critic is a new Value network. It evaluates how good the state is ($V_\phi(s)$).
class ActorCritic(nn.Module):
    def __init__(self, obs_space_size, act_space_size):
        super().__init__()

        self.obs_space_size = obs_space_size
        self.act_space_size = act_space_size

        # Predicts the mean of the actions
        self.actor=nn.Sequential(
            nn.Linear(self.obs_space_size,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,self.act_space_size),
            nn.Tanh()
        )

        self.std = nn.Parameter(torch.zeros(1, act_space_size))

        # Takes the state and predicts a single scalar value V(s)
        self.critic=nn.Sequential(
            nn.Linear(self.obs_space_size,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output is 1 scalar
        )

    def act_and_critic(self,state,device):
      state=torch.from_numpy(state).float().unsqueeze(0).to(device)
      mean=self.actor(state)
      std=self.std.exp().expand_as(mean)

      m=Normal(mean,std)
      action=m.sample()

      log_prob=m.log_prob(action).sum(dim=-1)
      entropy=m.entropy().sum(dim=-1)

      value=self.critic(state)

      action_np=action.detach().cpu().numpy()[0]

      return action_np, log_prob, value, entropy
