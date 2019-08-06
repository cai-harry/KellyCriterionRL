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
    # Q-values for each [state, action] pair
    _Q: np.ndarray = None

    # Number of times each [state, action] pair has been visited
    _N: np.ndarray = None

    def __init__(self, env, load_from_directory: Path = None, load_N: bool = True):
        self._Q = np.zeros(shape=(env.NUM_STATES, env.NUM_ACTIONS))
        self._N = np.zeros_like(self._Q)
        if load_from_directory is not None:
            self.load_parameters(load_from_directory, load_N)

    def get_action(self,
                   state_idx: int,
                   explore_probability: float = 0):
        explore = np.random.random() < explore_probability
        if explore:
            action = np.random.randint(0, state_idx + 1)
        else:  # exploit
            action = np.argmax(self._Q[state_idx, :state_idx + 1])

        return action

    def train(self,
              env, num_episodes,
              epsilons_each_episode=None,
              exploring_start=False,
              plot_training_rewards=True,
              use_tensorboard=False):

        if epsilons_each_episode is None:
            # use harmonic epsilon decay by default
            epsilons_each_episode = np.ones_like(num_episodes) / (np.arange(num_episodes) + 1)

        logger = TrainingLogger(num_episodes, epsilons_each_episode, plot_training_rewards, use_tensorboard)

        for episode_idx in range(num_episodes):

            # record a single episode
            env.reset(random_starting_money=exploring_start)
            finished = False
            states_this_episode = [env.get_state()]  # should be just the starting state
            actions_this_episode = []
            rewards_this_episode = []
            epsilon = epsilons_each_episode[episode_idx]
            while not finished:
                state_before_step = states_this_episode[-1]
                action = self.get_action(state_before_step, epsilon)
                actions_this_episode.append(action)
                new_state, reward, finished, debug_info = env.step(action)
                rewards_this_episode.append(reward)
                states_this_episode.append(new_state)
            total_reward_this_episode = sum(rewards_this_episode)

            logger.on_episode_end(episode_idx, total_reward_this_episode, epsilon)

            # Monte Carlo control: update N and Q arrays
            for s, a in zip(states_this_episode, actions_this_episode):
                self._N[s, a] += 1
                delta = (total_reward_this_episode - self._Q[s, a]) / self._N[s, a]
                self._Q[s, a] += delta

        logger.on_training_end()

    def plot_policy(self, optimal_policy=None):
        policy = [self.get_action(s, explore_probability=0) for s in range(250)]
        plt.plot(policy, label="Learned policy")
        plt.title("Learned policy (exploit mode, ie. epsilon=0)")
        plt.xlabel("Bankroll ($)")
        plt.ylabel("Bet size ($)")
        if optimal_policy is not None:
            plt.plot(optimal_policy, label="Optimal behaviour (Kelly Criterion)")
        plt.legend()
        plt.show()

    def plot_Q_values(self):
        # plot Q values
        plt.imshow(self._Q)
        plt.colorbar()
        plt.title("Q values heatmap")
        plt.xlabel("Action, ie. bet size ($)")
        plt.ylabel("State, ie. bankroll ($)")
        plt.show()

    def plot_N_values(self):
        # plot number of times each state is visited
        plt.imshow(np.log10(self._N))
        plt.colorbar()
        plt.title("N values heatmap (displaying log10(N) instead of raw N)")
        plt.xlabel("Action, ie. bet size ($)")
        plt.ylabel("State, ie. bankroll ($)")
        plt.show()

    def save_parameters(self, to_directory: Path = MODELS_DIR / "latest"):
        # make save directory if it doesn't exist
        to_directory.mkdir(parents=True, exist_ok=True)
        # save in created directory csv's
        np.savetxt(to_directory / "Q.csv", self._Q, fmt='%.2f')
        np.savetxt(to_directory / "N.csv", self._N, fmt='%d')

    def load_parameters(self, from_directory: Path, load_N: bool):
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
                 total_rewards_running_mean_N: int = None
                 ):
        self.num_episodes = num_episodes
        self.plot_total_rewards = plot_total_rewards
        self.use_tensorboard = use_tensorboard
        self.epsilons = epsilons_each_episode

        self._total_rewards = []

        self._total_rewards_running_mean = []
        if total_rewards_running_mean_N is not None:
            self._running_mean_N = total_rewards_running_mean_N
        else:
            self._running_mean_N = max(1, num_episodes // 100)

        self._tqdm_progress_bar = tqdm(total=num_episodes)

        self._tensorboard_writer = tensorboardX.SummaryWriter(
                log_dir=LOGS_DIR/strftime("%Y-%m-%d@%H-%M-%S"))

    def on_episode_end(self, episode_idx, total_reward_this_episode, epsilon_this_episode):
        self._total_rewards.append(total_reward_this_episode)

        mean_of_last_N = np.mean(
            self._total_rewards[-self._running_mean_N:]
        )
        self._total_rewards_running_mean.append(mean_of_last_N)

        self._tqdm_progress_bar.update(1)

        self._tensorboard_writer.add_scalar("Total Episode Reward", total_reward_this_episode, episode_idx)


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
