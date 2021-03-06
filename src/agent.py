from pathlib import Path
from time import strftime

import matplotlib.pyplot as plt
import numpy as np
import tensorboardX
from tqdm.auto import tqdm

from .context import MODELS_DIR, LOGS_DIR
from .environment import Environment

plt.style.use("ggplot")


class Agent:
    """
    The RL agent that should hopefully learn to navigate the coin flipping environment.

    Currently uses model-free Monte Carlo learning.
    """

    # Q-values for each [state, action] pair
    _Q: np.ndarray = None

    # Number of times each [state, action] pair has been visited
    _N: np.ndarray = None

    def __init__(self, env: Environment, load_from_directory: Path = None, load_N: bool = True):
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        self._Q = np.zeros(shape=(num_states, num_actions))
        self._N = np.zeros_like(self._Q)
        if load_from_directory is not None:
            self.load_parameters(load_from_directory, load_N)

    def act(self,
            state_idx: int,
            explore_probability: float = 0) -> int:
        # TODO: hacky. should use env.get_legal_actions()
        explore = np.random.random() < explore_probability
        if explore:
            action = np.random.randint(0, state_idx + 1)
        else:  # exploit
            action = np.argmax(self._Q[state_idx, :state_idx + 1])

        return action

    def train(self,
              env: Environment,
              num_episodes: int,
              epsilons_each_episode: np.ndarray = None,
              exploring_start: bool = False,
              plot_training_rewards: bool = True,
              use_tensorboard: bool = False):

        """
        Train the agent over the specified number of episodes.

        :param env: The environment whose rewards the agent should maximise
        :param num_episodes: The number of episodes to run
        :param epsilons_each_episode: The exploration probability for each episode.
            Uses linear decay by default.
        :param exploring_start: Whether to reset the environment to a different starting state each time.
        :param plot_training_rewards: Whether to make a plot of rewards after training is complete.
        :param use_tensorboard: Whether to record rewards and other stats to Tensorboard.
        :return:
        """

        if epsilons_each_episode is None:
            # linear decay by default
            epsilons_each_episode = np.linspace(start=1, stop=0, num=num_episodes)

        logger = TrainingLogger(num_episodes, epsilons_each_episode, plot_training_rewards, use_tensorboard)

        try:
            for episode_idx, epsilon in zip(range(num_episodes), epsilons_each_episode):
                actions, states, rewards = self._run_episode(
                    env, epsilon, exploring_start)
                self._update_parameters(actions, states, rewards)
                logger.on_episode_end(episode_idx, rewards, epsilon)

        except KeyboardInterrupt:
            logger.print(f"Ended early at episode {episode_idx} due to KeyboardInterrupt")

        # save N and Q values to the default directory for the latest trained model
        self.save_parameters()

        logger.on_training_end()

    def _run_episode(self, env: Environment, epsilon: float, exploring_start: bool):
        """Record a single episode of experience"""
        env.reset(random_starting_money=exploring_start)
        finished = False
        states_this_episode = [env.get_state()]  # should be just the starting state
        actions_this_episode = []
        rewards_this_episode = []
        while not finished:
            state_before_step = states_this_episode[-1]
            action = self.act(state_before_step, epsilon)
            actions_this_episode.append(action)
            new_state, reward, finished, debug_info = env.step(action)
            rewards_this_episode.append(reward)
            states_this_episode.append(new_state)
        total_reward_this_episode = sum(rewards_this_episode)
        return actions_this_episode, states_this_episode, total_reward_this_episode

    def _update_parameters(self, episode_actions, episode_states, episode_total_reward):
        """Monte Carlo control: update N and Q arrays"""
        for s, a in zip(episode_states, episode_actions):
            self._N[s, a] += 1
            delta = (episode_total_reward - self._Q[s, a]) / self._N[s, a]
            self._Q[s, a] += delta

    def plot_policy(self, optimal_policy=None) -> ():
        policy = [self.act(s, explore_probability=0) for s in range(250)]
        plt.plot(policy, label="Learned policy")
        plt.title("Learned policy (exploit mode, ie. epsilon=0)")
        plt.xlabel("Bankroll ($)")
        plt.ylabel("Bet size ($)")
        if optimal_policy is not None:
            plt.plot(optimal_policy, label="Optimal behaviour (Kelly Criterion)")
        plt.legend()
        plt.show()

    def plot_Q_values(self) -> ():
        # plot Q values
        plt.imshow(self._Q)
        plt.colorbar()
        plt.title("Q values heatmap")
        plt.xlabel("Action, ie. bet size ($)")
        plt.ylabel("State, ie. bankroll ($)")
        plt.show()

    def plot_N_values(self) -> ():
        # plot number of times each state is visited
        plt.imshow(np.log10(self._N))
        plt.colorbar()
        plt.title("N values heatmap (displaying log10(N) instead of raw N)")
        plt.xlabel("Action, ie. bet size ($)")
        plt.ylabel("State, ie. bankroll ($)")
        plt.show()

    def save_parameters(self, to_directory: Path = MODELS_DIR / "latest") -> ():
        """
        Save N and Q values to the given directory.
        """
        # make save directory if it doesn't exist
        to_directory.mkdir(parents=True, exist_ok=True)
        # save in created directory csv's
        np.savetxt(to_directory / "Q.csv", self._Q, fmt='%.2f')
        np.savetxt(to_directory / "N.csv", self._N, fmt='%d')

    def load_parameters(self, from_directory: Path, load_N: bool) -> ():
        """
        Load N and Q values from the given directory.
        """
        self._Q = np.loadtxt(from_directory / "Q.csv")
        if load_N:
            self._N = np.loadtxt(from_directory / "N.csv")


class TrainingLogger:
    """
    Contains code for logging training progress and statistics.

    Kept in a separate class to reduce code bloat in Agent.train()
    """

    def __init__(self, num_episodes: int,
                 epsilons_each_episode: np.ndarray,
                 plot_total_rewards: bool,
                 use_tensorboard: bool,
                 total_rewards_running_mean_N: int = 3
                 ):
        self.num_episodes = num_episodes
        self.plot_total_rewards = plot_total_rewards
        self.using_tensorboard = use_tensorboard
        self.epsilons = epsilons_each_episode

        self._total_rewards = []

        self._total_rewards_running_mean = []
        self._running_mean_N = total_rewards_running_mean_N

        self._tqdm_progress_bar = tqdm(total=num_episodes)

        if use_tensorboard:
            self._tensorboard_writer = tensorboardX.SummaryWriter(
                log_dir=LOGS_DIR / strftime("%Y-%m-%d@%H-%M-%S"))

    def print(self, msg: str):
        self._tqdm_progress_bar.write(msg)

    def on_episode_end(self, episode_idx: int, total_reward_this_episode: float, epsilon_this_episode: float):
        self._total_rewards.append(total_reward_this_episode)

        mean_of_last_N = np.mean(
            self._total_rewards[-self._running_mean_N:]
        )
        self._total_rewards_running_mean.append(mean_of_last_N)

        self._tqdm_progress_bar.update(1)

        if self.using_tensorboard:
            self._tensorboard_writer.add_scalar("Total Episode Reward", total_reward_this_episode, episode_idx)
            self._tensorboard_writer.add_scalar("Epsilon", epsilon_this_episode, episode_idx)

    def on_training_end(self):
        if self.plot_total_rewards:
            self._plot_total_rewards()
        self._tqdm_progress_bar.close()

    def _plot_total_rewards(self):
        plt.plot(self._total_rewards_running_mean,
                 label=f"Total rewards (running mean with N={self._running_mean_N})")

        # plot epsilon decay curve
        if self.epsilons is not None:
            plt.plot(self.epsilons * np.max(self._total_rewards_running_mean),
                     label="Epsilon decay curve (not to scale)")
            plt.title("Rewards during training")
            plt.xlabel("Episode")
            plt.ylabel("Reward ($)")
            plt.legend()
            plt.show()
