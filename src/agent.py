import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .environment import Environment

plt.style.use("ggplot")


class Agent:
    # Q-values for each [state, action] pair
    _Q: np.ndarray = None

    # Number of times each [state, action] pair has been visited
    _N: np.ndarray = None

    def __init__(self, env: Environment, Q_values: np.ndarray = None):
        if Q_values is not None:
            self._Q = Q_values
        else:
            # TODO: this makes assumptions about env
            self._Q = np.zeros(shape=(env.NUM_STATES, env.NUM_ACTIONS))
        self._N = np.zeros_like(self._Q)

    def get_action(self,
                   state_idx: int,
                   explore_probability: float = 0):
        explore = np.random.random() < explore_probability
        if explore:
            action = np.random.randint(0, state_idx + 1)
        else:  # exploit
            action = np.argmax(self._Q[state_idx, :state_idx + 1])

        return action

    def train(self, env, episodes, epsilons_each_episode=None, exploring_start=False):

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

        np.savetxt("models/Q.csv", self._Q, fmt='%.2f')
        np.savetxt("models/N.csv", self._N, fmt='%d')

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
