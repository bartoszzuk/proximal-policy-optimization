import random
from abc import abstractmethod, ABC

import torch
from torch import Tensor
from typing_extensions import Self

from dqn.config import DQNConfig


class PolicySelection(ABC):

    @abstractmethod
    def __call__(self, values: Tensor) -> int:
        raise NotImplementedError()

    @abstractmethod
    def update(self) -> None:
        raise NotImplementedError()


class EpsilonGreedy(PolicySelection):

    @classmethod
    def from_config(cls, config: DQNConfig) -> Self:
        return cls(config.epsilon_start, config.epsilon_end, config.epsilon_decay)

    def __init__(self, start: float, end: float, decay: float) -> None:
        self.start = start
        self.end = end
        self.decay = decay
        self.step = 0

    def __call__(self, values: Tensor) -> int:
        epsilon = self.start * self.decay ** self.step
        epsilon = max(epsilon, self.end)

        if random.random() > epsilon:
            action = torch.argmax(values).item()
        else:
            actions = values.shape[-1]
            action = random.randint(0, actions - 1)

        return action

    def update(self) -> None:
        self.step += 1
