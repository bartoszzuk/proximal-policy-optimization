from abc import ABC
from typing import Literal

from torch import nn

Mode = Literal['train', 'eval']


class Trainable(ABC):

    def __init__(self) -> None:
        self.mode: Mode = 'train'

    def train(self) -> None:
        # Change mode of all trainable attributes
        for attribute, value in vars(self).items():
            if isinstance(value, Trainable | nn.Module):
                value.train()

        self.mode = 'train'

    def eval(self) -> None:
        # Change mode of all trainable attributes
        for attribute, value in vars(self).items():
            if isinstance(value, Trainable | nn.Module):
                value.eval()

        self.mode = 'eval'
