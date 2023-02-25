from torch import nn, Tensor


class DQN(nn.Module):

    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, observation: Tensor) -> Tensor:
        return self.layers(observation)
