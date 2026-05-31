import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import csv

from src.config import MODEL_PATH

def train_actor_critic(env,model,device,n_iterations,num_steps,gamma,lr, print_every=10, model_save_path=MODEL_PATH, csv_save_path="output/training.csv"):
  optimizer=optim.Adam(model.parameters(),lr=lr)
  scores=[]
  durations=[]
  total_env_steps=0
  best_avg_reward=-float('inf')

  os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
  with open(csv_save_path, mode='w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(["update_step", "total_episodes", "total_env_steps", "average_reward", "average_episode_duration", "average_policy_entropy"])

  state,_=env.reset()
  episode_reward=0
  episode_duration=0

  for update in range(1,n_iterations+1):
    log_probs,values,rewards,dones,entropies=[],[],[],[],[]

    for _ in range(num_steps):
      action,log_prob,value,entropy=model.act_and_critic(state,device)
      next_state,reward,terminated,truncated,_=env.step(action)
      total_env_steps+=1
      done=terminated or truncated

      log_probs.append(log_prob)
      values.append(value)
      rewards.append(reward)
      dones.append(done)
      entropies.append(entropy.item())

      episode_reward+=reward
      episode_duration+=1

      if done:
        scores.append(episode_reward)
        durations.append(episode_duration)
        state,_=env.reset()
        episode_reward=0
        episode_duration=0
      else:
        state=next_state

    #made a step, now bootstrap for this step.

    with torch.no_grad():
      _,_,next_value,_=model.act_and_critic(state,device)
      next_value=next_value.squeeze().item() if not done else 0.0

    returns=[]
    advantages=[]
    R=next_value

    for r,v,is_done in zip(reversed(rewards),reversed(values),reversed(dones)):
      R=r+gamma*R*(1.0-is_done)
      returns.insert(0,R)

      advantage=R-v.item()
      advantages.insert(0,advantage)

    log_probs_tensor=torch.stack(log_probs).squeeze()

    values_tensor=torch.cat(values).squeeze()
    returns_tensor=torch.tensor(returns,dtype=torch.float32,device=device)
    advantages_tensor=torch.tensor(advantages,dtype=torch.float32,device=device)

    advantages_tensor=(advantages_tensor-advantages_tensor.mean())/(advantages_tensor.std()+1e-8)

    actor_loss=-(log_probs_tensor*advantages_tensor).mean()
    critic_loss=F.mse_loss(values_tensor,returns_tensor)

    total_loss=actor_loss+critic_loss

    optimizer.zero_grad()
    total_loss.backward()

    optimizer.step()

    avg_reward=np.mean(scores[-100:]) if scores else 0.0
    avg_duration=np.mean(durations[-100:]) if durations else 0.0
    avg_entropy=np.mean(entropies)

    with open(csv_save_path, mode='a',newline='') as f:
        writer=csv.writer(f)
        writer.writerow([update, len(scores), total_env_steps, avg_reward, avg_duration, avg_entropy])

    if update%print_every==0:
      print(f"Update {update} | Avg Reward (last 100): {avg_reward:.1f} | Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f}")
    
    if len(scores)>=100 and avg_reward>best_avg_reward:
      best_avg_reward=avg_reward
      torch.save(model.state_dict(), model_save_path)
      print(f"New best model saved to {model_save_path} (avg reward: {avg_reward:.1f})")
  return scores
