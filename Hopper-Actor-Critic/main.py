import gymnasium as gym
import torch

from src.generate_video import generate_video
from src.model import ActorCritic
from src.train import train_actor_critic
from src.config import (
    ENV_NAME,
    N_ITERATIONS,
    GAMMA,
    LR,
    PRINT_EVERY,

    MODEL_PATH,
    CSV_PATH,
    NUM_STEPS,
)

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_dir = __import__("pathlib").Path(MODEL_PATH).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    env=gym.make(ENV_NAME)

    obs_space_size=env.observation_space.shape[0]
    act_space_size=env.action_space.shape[0]

    ac_model=ActorCritic(obs_space_size,act_space_size).to(device)

    print(f"Training on {device}...")
    scores=train_actor_critic(
        env=env,
        model=ac_model,
        device=device,
        n_iterations=N_ITERATIONS,
        num_steps=NUM_STEPS,
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
