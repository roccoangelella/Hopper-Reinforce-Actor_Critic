import gymnasium as gym
import torch

from src.generate_video import generate_video
from src.model import Policy
from src.train import reinforce
from src.config import (
    ENV_NAME,
    N_ITERATIONS,
    BATCH_SIZE,
    GAMMA,
    LR,
    PRINT_EVERY,
    MODEL_PATH,
    CSV_PATH,
)

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env=gym.make(ENV_NAME)

    obs_space=env.observation_space
    act_space=env.action_space

    policy=Policy(obs_space.shape[0],act_space.shape[0]).to(device)

    print("Starting training...")
    scores=reinforce(
        policy,
        env,
        device,
        n_iterations=N_ITERATIONS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        lr=LR,
        print_every=PRINT_EVERY,
        model_save_path=MODEL_PATH,
        csv_save_path=CSV_PATH
    )

    env.close()

    print("Training finished. Recording video of a full episode...")
    generate_video()

if __name__=="__main__":
    main()
