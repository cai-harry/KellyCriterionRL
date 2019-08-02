# %% Setup
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

plt.style.use("ggplot")


# %% Environment

class Environment:
    _win_probability = None
    _starting_money = None
    _max_bets = None
    _player_money = None
    _bets_remaining = None
    _finished = None

    def __init__(self, win_probability: float = 0.6, starting_money: float = 25, max_bets: int = 300,
                 print_msgs: bool = False):
        self._win_probability = win_probability
        self._starting_money = starting_money
        self._max_bets = max_bets
        self.reset()

    def get_state(self) -> float:
        return np.round(self._player_money)

    def reset(self):
        self._player_money = self._starting_money
        self._bets_remaining = self._max_bets
        self._finished = False

    def step(self, bet_size: float, print_msgs: bool = False) -> float:

        if self._finished:
            raise AttributeError("Game has finished")
        if bet_size > self._player_money:
            raise ValueError(f"Cannot bet ${bet_size} when player has ${self._player_money}")

        if print_msgs:
            print(
                f"Player 1 bets ${bet_size} from a bankroll of ${self._player_money} ({np.round(100 * bet_size / self._player_money)}%)")

        player_money_before_bet = self._player_money

        if np.random.random() < self._win_probability:
            self._player_money = min(player_money_before_bet + bet_size, 250)
            if print_msgs:
                print(f"Player 1 WINS the bet")
        else:
            self._player_money = max(player_money_before_bet - bet_size, 0)
            if print_msgs:
                print(f"Player 1 LOSES the bet")

        self._bets_remaining -= 1

        if self._player_money <= 0 or self._bets_remaining <= 0:
            self._finished = True

        if self._player_money >= 250:
            self._player_money = 250
            self._finished = True

        return self._player_money, self._player_money - player_money_before_bet, self._finished

    def print_summary(self):
        print(f"STATE\n"
              f"\tMoney: ${self._player_money}\n"
              f"\tBets remaining: {self._bets_remaining}\n"
              )


def test_environment():
    # test for Environment
    wins = 0
    busts = 0
    total_payout = 0
    num_episodes = 1000

    for i in range(num_episodes):
        env = Environment()
        finished = False
        while not finished:
            money = env.get_state()
            money, reward, finished = env.step(bet_size=np.round(money * 0.2))
            env.print_summary()

        final_money = env.get_state()
        total_payout += final_money
        if final_money >= 250:
            print("Player 1 reached maximum")
            wins += 1
        if final_money <= 0:
            print("Player 1 went bust")
            busts += 1

    print()
    print(f"Player 1 reached maximum {wins} times of {num_episodes}, and went bust {busts} times")
    print(f"The average payout was ${total_payout / num_episodes}")


test_environment()


# %% Agent
class Agent:
    _Q: np.ndarray = None

    def __init__(self):
        # todo: this makes assumptions about env
        self._Q = np.random.random_integers(low=-250, high=250, size=(251, 250))

    def train(self, env, start_epsilon, episodes):
        total_rewards = []
        N = np.zeros_like(self._Q)  # array recording number of times each Q-value has been visited
        for episode_idx in range(episodes):

            # record a single episode
            env.reset()
            finished = False
            states_this_episode = [env.get_state()]  # should be just the starting state
            actions_this_episode = []
            rewards_this_episode = []
            epsilon = start_epsilon - (episode_idx / episodes) * start_epsilon
            while not finished:
                state_before_step = states_this_episode[-1]
                action = self.get_action(state_before_step, epsilon)
                actions_this_episode.append(action)
                new_state, reward, finished = env.step(action)
                states_this_episode.append(new_state)
                rewards_this_episode.append(reward)
            total_reward_this_episode = sum(rewards_this_episode)
            print(f"Total reward for episode {episode_idx + 1}/{episodes}: {total_reward_this_episode}")
            total_rewards.append(total_reward_this_episode)

            # Monte Carlo control: update N and Q arrays
            for s, a in zip(states_this_episode, actions_this_episode):
                N[s, a] += 1
                delta = (total_reward_this_episode - self._Q[s, a]) / N[s, a]
                self._Q[s, a] += delta

        # print(f"Training finished")
        return total_rewards

    def get_action(self, state_idx, explore_probability):
        explore = np.random.random() < explore_probability
        if explore:
            action = np.random.randint(0, state_idx)
        else:  # exploit
            action = np.argmax(self._Q[state_idx, :state_idx])

        return action


def test_agent(num_episodes):
    env = Environment()
    agent = Agent()
    total_rewards = agent.train(env, start_epsilon=1, episodes=num_episodes)
    total_rewards_running_mean = np.convolve(total_rewards, np.ones(20)/20, mode='valid')
    plt.plot(total_rewards_running_mean)
    plt.plot(np.linspace(0.8*250, 0, num_episodes))
    plt.show()
    np.savetxt("Q.csv", agent._Q, fmt='%.2f')
    plt.imshow(agent._Q)
    plt.show()


test_agent(num_episodes=100000)
