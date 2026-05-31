import os
from pathlib import Path

if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
    render_device = Path("/dev/dri/renderD128")
    # Prefer software rendering when there's no X server and the GPU render node is unavailable.
    os.environ["MUJOCO_GL"] = "egl" if render_device.exists() and os.access(render_device, os.R_OK) else "osmesa"

import gymnasium as gym
import imageio
import torch

from src.config import ENV_NAME, MODEL_PATH, VIDEO_FOLDER
from src.model import ActorCritic


VIDEO_FPS = 60


def generate_video():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = Path(MODEL_PATH)
    video_path = Path(VIDEO_FOLDER)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing policy weights: {model_path}")

    video_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    model = ActorCritic(
        env.observation_space.shape[0],
        env.action_space.shape[0],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    frames = []
    state, _ = env.reset()
    done = False

    with torch.no_grad():
        while not done:
            frames.append(env.render())
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action = model.actor(state_tensor).squeeze(0).cpu().numpy()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()

    imageio.mimwrite(video_path, frames, fps=VIDEO_FPS)
    print(f"Video saved to {video_path}")


def main():
    generate_video()


if __name__ == "__main__":
    main()