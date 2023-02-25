from dataclasses import dataclass
from pathlib import Path

import toml
from typing_extensions import Self


@dataclass(frozen=True)
class DQNConfig:
    episodes: int
    max_steps: int
    replay_size: int
    batch_size: int
    learning_rate: float
    epsilon_decay: float
    epsilon_start: float
    epsilon_end: float
    tau: float
    gamma: float
    seed: int
    observation_dim: int
    action_dim: int

    @classmethod
    def from_toml(cls, path: Path) -> Self:
        with open(path, 'r') as file:
            config = toml.load(file)
            config = config[cls.__name__]

        return cls(**config)
