import random
from dataclasses import dataclass, asdict
from typing import SupportsFloat

import numpy as np
import torch
from torch import Tensor
from torch.types import Device
from typing_extensions import Self


@dataclass(frozen=True)
class Experience:
    observation: np.ndarray
    next_observation: np.ndarray
    action: int
    reward: float | SupportsFloat
    done: bool


@dataclass(frozen=True)
class ExperienceBatch:
    observations: Tensor
    next_observations: Tensor
    actions: Tensor
    rewards: Tensor
    done: Tensor

    def to(self, device: Device) -> Self:
        return ExperienceBatch(**{name: tensor.to(device) for name, tensor in asdict(self).items()})


class ExperienceReplay:

    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer = []

    def add(self, experience: Experience) -> None:
        if len(self.buffer) == self.size:
            self.buffer.pop(0)

        self.buffer.append(experience)

    def sample(self, size: int) -> ExperienceBatch:
        size = min(len(self.buffer), size)

        experiences = random.sample(self.buffer, k=size)

        observations = np.vstack([experience.observation for experience in experiences])
        observations = torch.from_numpy(observations)

        next_observations = np.vstack([experience.next_observation for experience in experiences])
        next_observations = torch.from_numpy(next_observations)

        actions = torch.tensor([experience.action for experience in experiences], dtype=torch.int64)
        rewards = torch.tensor([experience.reward for experience in experiences], dtype=torch.float)
        done = torch.tensor([experience.done for experience in experiences], dtype=torch.int)

        return ExperienceBatch(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            done=done
        )

    def __len__(self) -> int:
        return len(self.buffer)

