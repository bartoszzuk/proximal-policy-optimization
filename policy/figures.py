import numpy as np
from matplotlib import pyplot as plt

from policy.dqn.trainer import DQNTrainer
from matplotlib.colors import get_named_colors_mapping


COLORS = get_named_colors_mapping()


def plot_recent_mean_scores(trainer: DQNTrainer) -> None:
    scores = trainer.recent_mean_score
    episodes = np.arange(len(scores))

    plt.figure(figsize=(10, 8))

    plt.title('Mean Score from recent 100 episodes')
    plt.xlabel('Episode')
    plt.ylabel('Mean Score')

    plt.plot(episodes, scores, color=COLORS['tab:blue'], label='DQN')
    plt.savefig('out/recent-mean-scores.png')


def plot_recent_mean_steps(trainer: DQNTrainer) -> None:
    steps = trainer.recent_mean_steps
    episodes = np.arange(len(steps))

    plt.figure(figsize=(10, 8))

    plt.title('Mean Steps from recent 100 episodes')
    plt.xlabel('Episode')
    plt.ylabel('Mean Steps')

    plt.plot(episodes, steps, color=COLORS['tab:orange'], label='DQN')
    plt.savefig('out/recent-mean-steps.png')

