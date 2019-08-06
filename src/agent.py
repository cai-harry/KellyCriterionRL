from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from .context import MODELS_DIR
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

    def train(self, env, episodes, epsilons_each_episode=None, exploring_start=False, plot_training_rewards=True):

        if epsilons_each_episode is None:
            # use harmonic epsilon decay by default
            epsilons_each_episode = np.ones_like(episodes) / (np.arange(len(episodes)) + 1)

        total_rewards = []
        for episode_idx in tqdm(range(episodes)):

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
                states_this_episode.append(new_state)
                rewards_this_episode.append(reward)
            total_reward_this_episode = sum(rewards_this_episode)
            total_rewards.append(total_reward_this_episode)

            # Monte Carlo control: update N and Q arrays
            for s, a in zip(states_this_episode, actions_this_episode):
                self._N[s, a] += 1
                delta = (total_reward_this_episode - self._Q[s, a]) / self._N[s, a]
                self._Q[s, a] += delta

        if plot_training_rewards:
            plot_rewards_during_training(total_rewards, epsilons_each_episode)

        return total_rewards

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


def plot_rewards_during_training(total_rewards: np.ndarray, epsilons: np.ndarray):
    # plot running mean of rewards during training
    running_mean_N = max(1, len(total_rewards) // 100)
    total_rewards_running_mean = np.convolve(total_rewards, np.ones(running_mean_N) / running_mean_N, mode='valid')
    plt.plot(total_rewards_running_mean, label="Total rewards (running mean with N=100)")

    # plot epsilon decay curve
    plt.plot(epsilons * np.max(total_rewards_running_mean), label="Epsilon decay curve (not to scale)")
    plt.title("Rewards during training")
    plt.xlabel("Episode")
    plt.ylabel("Reward ($)")
    plt.legend()
    plt.show()
