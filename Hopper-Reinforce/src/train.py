import numpy as np
import torch
from torch import optim
import os
import csv

def compute_returns(rewards,gamma):
    returns=[]
    G=0

    for r in reversed(rewards):
        G=r+gamma*G
        returns.insert(0,G)

    returns=torch.tensor(returns,dtype=torch.float32)
    return returns

def reinforce(
    policy,
    env,
    device,
    n_iterations=100000,
    batch_size=10,
    gamma=0.99,
    lr=3e-4,
    print_every=10,
    model_save_path="hopper_policy.pth",
    csv_save_path="output/training.csv"
):
    optimizer=optim.Adam(policy.parameters(),lr=lr)
    scores=[]
    total_episodes_played=0
    total_env_steps=0
    best_avg_reward=-float('inf')

    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    with open(csv_save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["update_step", "total_episodes", "total_env_steps", "average_reward", "average_episode_duration", "average_policy_entropy"])

    for iteration in range(1,n_iterations+1):

        batch_loss=[]
        batch_rewards=[]
        batch_durations=[]
        batch_entropies=[]
        batch_log_probs=[]
        batch_returns=[]

        for _ in range(batch_size): #we run batches of episodes to avoid updating weights on a single episode, which would be too noisy cosidering how dumb reinforce is for this task
            state,_=env.reset()
            rewards=[]
            log_probs=[]
            entropies=[]
            done=False
            duration=0

            while not done:
                action,log_prob,entropy=policy.act(state,device)

                state,reward,terminated,truncated,_=env.step(action)
                done=terminated or truncated

                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy.item())
                duration+=1
                total_env_steps+=1

            returns=compute_returns(rewards,gamma)
            batch_log_probs.append(log_probs)
            batch_returns.append(returns)
            batch_rewards.append(sum(rewards))
            batch_durations.append(duration)
            batch_entropies.append(sum(entropies)/len(entropies))
            total_episodes_played+=1
            scores.append(sum(rewards))

        all_returns=torch.cat(batch_returns)
        returns_mean=all_returns.mean()
        returns_std=all_returns.std()+1e-9

        for log_probs,returns in zip(batch_log_probs,batch_returns):
            returns=(returns-returns_mean)/returns_std
            episode_loss=[]
            for log_prob,G in zip(log_probs,returns):
                episode_loss.append(-log_prob*G)
            batch_loss.append(torch.stack(episode_loss).sum())

        optimizer.zero_grad()

        loss=torch.stack(batch_loss).mean()
        loss.backward()

        optimizer.step()

        avg_reward=np.mean(scores[-100:]) if scores else 0.0
        avg_duration=np.mean(batch_durations)
        avg_entropy=np.mean(batch_entropies)

        with open(csv_save_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, total_episodes_played, total_env_steps, avg_reward, avg_duration, avg_entropy])

        if iteration%print_every==0:
            avg_score=np.mean(scores[-100:]) if len(scores)>=100 else np.mean(scores)
            print(
                f"Iteration {iteration}/{n_iterations} "
                f"(Total Episodes: {total_episodes_played}) | "
                f"Avg Reward (last 100): {avg_score:.1f}"
            )

        avg_score=np.mean(scores[-100:]) if len(scores)>=100 else np.mean(scores)
        if len(scores)>=100 and avg_score>best_avg_reward:
            best_avg_reward=avg_score
            torch.save(policy.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path} (avg reward: {avg_score:.1f})")

    return scores
