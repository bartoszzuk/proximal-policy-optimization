import random
from argparse import Namespace, ArgumentParser
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import torch
from gymnasium import wrappers

from policy.dqn import DQNAgent
from policy.dqn.config import DQNConfig
from policy.dqn import DQNTrainer


def parse_arguments(config: DQNConfig) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-e', '--episodes', type=int, default=config.episodes)
    parser.add_argument('-s', '--max-steps', type=int, default=config.max_steps)
    parser.add_argument('-r', '--replay-size', type=int, default=config.replay_size)
    parser.add_argument('-b', '--batch-size', type=int, default=config.batch_size)
    parser.add_argument('-l', '--learning-rate', type=float, default=config.learning_rate)
    parser.add_argument('-d', '--epsilon-decay', type=float, default=config.epsilon_decay)
    parser.add_argument('-x', '--epsilon-start', type=float, default=config.epsilon_start)
    parser.add_argument('-y', '--epsilon-end', type=float, default=config.epsilon_end)
    parser.add_argument('-t', '--tau', type=float, default=config.tau)
    parser.add_argument('-g', '--gamma', type=float, default=config.gamma)
    parser.add_argument('-z', '--seed', type=int, default=config.seed)

    return parser.parse_args()


def overwrite_config_with_arguments(config: DQNConfig) -> DQNConfig:
    arguments = parse_arguments(config)
    arguments = vars(arguments)

    overwritten = asdict(config) | arguments

    return DQNConfig(**overwritten)


def ensure_reproducibility(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simple_episode_trigger(episode: int) -> bool:
    return episode % 100 == 0


def train() -> None:
    path = Path('config.toml').absolute()

    config = DQNConfig.from_toml(path)
    config = overwrite_config_with_arguments(config)

    ensure_reproducibility(config.seed)

    environment = gym.make("LunarLander-v2", max_episode_steps=config.max_steps, render_mode='rgb_array')
    environment = wrappers.RecordEpisodeStatistics(environment)
    environment = wrappers.RecordVideo(
        env=environment,
        video_folder='out',
        episode_trigger=simple_episode_trigger,
        name_prefix='lunar-video',
        disable_logger=True
    )

    agent = DQNAgent(config)

    trainer = DQNTrainer(agent, config)
    trainer.fit(environment)

    environment.close()


if __name__ == '__main__':
    train()
