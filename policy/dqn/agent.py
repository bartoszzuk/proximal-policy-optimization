import numpy as np
import torch
from torch.nn import functional
from torch.optim import Adam

from policy.dqn.config import DQNConfig
from policy.dqn.model import DQN
from policy.dqn.replay import ExperienceBatch
from policy.dqn.selection import EpsilonGreedy
from policy.helpers import Trainable

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DQNAgent(Trainable):

    def __init__(self, config: DQNConfig) -> None:
        super().__init__()
        self.policy_network = DQN(config.observation_dim, config.action_dim).to(DEVICE)
        self.target_network = DQN(config.observation_dim, config.action_dim).to(DEVICE)
        self.config = config

        self.optimizer = Adam(self.policy_network.parameters(), lr=config.learning_rate)
        self.exploration = EpsilonGreedy.from_config(config)

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> int:
        observation = torch.from_numpy(observation)
        observation = torch.unsqueeze(observation, dim=0)
        observation = observation.to(DEVICE)

        values = self.policy_network(observation)
        values = values.detach()

        # Policy selection e.g. Epsilon Greedy
        return self.exploration(values)

    def update(self, batch: ExperienceBatch, gamma: float = None) -> None:
        batch = batch.to(DEVICE)
        gamma = gamma or self.config.gamma

        with torch.no_grad():
            targets = self.target_network(batch.next_observations)
            targets = targets.max(1).values
            targets = batch.rewards + gamma * targets * (1 - batch.done)
            targets = targets.unsqueeze(dim=1)

        predictions = self.policy_network(batch.observations)
        predictions = predictions.gather(1, batch.actions.unsqueeze(dim=1))

        loss = functional.huber_loss(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_network()

    def update_target_network(self, tau: float = None) -> None:
        tau = tau or self.config.tau

        policy_parameters = self.policy_network.parameters()
        target_parameters = self.target_network.parameters()

        for policy, target in zip(policy_parameters, target_parameters):
            combined = tau * policy.data + (1 - tau) * target.data
            target.data.copy_(combined)
