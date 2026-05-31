# Configuration hyperparameters for Hopper Actor-Critic

ENV_NAME = "Hopper-v5"

# Training parameters
N_ITERATIONS = 200000 # Starts learning much faster than REINFORCE
NUM_STEPS = 256
GAMMA = 0.99
LR = 1e-4 # Standard learning rate for Actor-Critic

# Logging and Saving
PRINT_EVERY = 10
MODEL_PATH = "output/policy.pth"
VIDEO_FOLDER = "output/run.mp4"
VIDEO_PREFIX = "eval_episode"
CSV_PATH = "output/training.csv"
