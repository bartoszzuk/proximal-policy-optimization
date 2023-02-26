import statistics
from typing import Literal

from gymnasium import Env, Wrapper

from policy.dqn.agent import DQNAgent
from policy.dqn.config import DQNConfig
from policy.dqn.replay import ExperienceReplay, Experience


def print_episode_statistics(environment: Env | Wrapper, episode: int) -> None:
    for attribute in ['return_queue', 'length_queue']:
        if not hasattr(environment, attribute):
            raise AttributeError(f'Attribute {attribute} not found, maybe RecordEpisodeStatistics missing')

    mean_score = statistics.mean(value.item() for value in environment.return_queue)
    mean_steps = statistics.mean(value.item() for value in environment.length_queue)

    end = '\n' if episode % 100 == 0 else ''

    print(f'\rEpisode {episode} | Mean Score {mean_score:.2f} | Mean Steps {mean_steps:.2f}', end=end)


class DQNTrainer:

    def __init__(self, agent: DQNAgent, config: DQNConfig) -> None:
        self.agent = agent
        self.config = config
        self.experiences = ExperienceReplay(config.replay_size)

    def fit(self, environment: Env) -> None:
        for episode in range(1, self.config.episodes + 1):
            self.run_episode(environment, stage='fit')
            self.agent.policy_selection.update()
            print_episode_statistics(environment, episode)

    def test(self, environment: Env) -> None:
        self.run_episode(environment, stage='test')

    def run_episode(self, environment: Env, stage: Literal['fit', 'test']) -> None:
        observation, _ = environment.reset(seed=self.config.seed)

        step, done = 0, False

        while not done:
            # Sample action from agent policy
            action = self.agent.act(observation)

            # Perform action and observe results
            next_observation, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated

            # For fit stage save observed experience and update agent
            if stage == 'fit':
                experience = Experience(observation, next_observation, action, reward, done)
                self.experiences.add(experience)
                self.update_agent_parameters(step)

            # Post step update
            observation = next_observation
            step += 1

    def is_update_ready(self, step: int) -> bool:
        return step % 4 == 0 and len(self.experiences) >= self.config.batch_size

    def update_agent_parameters(self, step: int) -> None:
        if self.is_update_ready(step):
            batch_size = self.config.batch_size
            batch = self.experiences.sample(batch_size)

            self.agent.update(batch)
