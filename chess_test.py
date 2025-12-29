import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

import gymnasium as gym
import chess_gym
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend

import matplotlib.pyplot as plt

# Create environment
env = gym.make("Chess-v0")
observation, info = env.reset()

# Create directory for frames
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

frame_idx = 0  # counter for filenames

terminal = False

while not terminal:
    # Plot and save the current frame
    fig, ax = plt.subplots()
    ax.imshow(observation)
    plt.axis('off')
    filename = os.path.join(frames_dir, f"frame_{frame_idx:03d}.png")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # close the figure to save memory
    print(f"Saved {filename}")
    frame_idx += 1

    # Take random action
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    terminal = terminated or truncated

env.close()
