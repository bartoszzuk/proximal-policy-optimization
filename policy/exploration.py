import random
from abc import abstractmethod, ABC

import torch
from torch import Tensor
from typing_extensions import Self

from policy.dqn.config import DQNConfig
from policy.helpers import Trainable


class Exploration(ABC):

    @abstractmethod
    def __call__(self, values: Tensor) -> int:
        raise NotImplementedError()

    @abstractmethod
    def update(self) -> None:
        raise NotImplementedError()


class EpsilonGreedy(Exploration, Trainable):

    @classmethod
    def from_config(cls, config: DQNConfig) -> Self:
        return cls(config.epsilon_start, config.epsilon_end, config.epsilon_decay)

    def __init__(self, start: float, end: float, decay: float) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.decay = decay
        self.step = 0

    def __call__(self, values: Tensor) -> int:
        if self.mode == 'train':
            epsilon = self.start * self.decay ** self.step
            epsilon = max(epsilon, self.end)
        else:
            epsilon = self.end

        if random.random() > epsilon:
            action = torch.argmax(values).item()
        else:
            actions = values.shape[-1]
            action = random.randint(0, actions - 1)

        return action

    def update(self) -> None:
        self.step += 1
