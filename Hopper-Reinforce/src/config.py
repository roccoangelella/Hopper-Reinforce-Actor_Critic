# Configuration hyperparameters for Hopper REINFORCE

ENV_NAME = "Hopper-v5"

# Training parameters
N_ITERATIONS = 200000
BATCH_SIZE = 10
GAMMA = 0.99
LR = 1e-4

# Logging and Saving
PRINT_EVERY = 10
MODEL_PATH = "output/policy.pth"
VIDEO_PATH = "output/run.mp4"
VIDEO_FPS = 60
CSV_PATH = "output/training.csv"
