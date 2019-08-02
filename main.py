# %% Setup
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

plt.style.use("ggplot")


# %% Environment

class Environment:
    WIN_PROBABILITY = None
    PAYOUT_RATIO = None
    STARTING_MONEY = None
    MAX_NUM_BETS = None
    MAX_MONEY = None

    _player_money = None
    _bets_remaining = None
    _finished = None

    def __init__(self,
                 win_probability: float = 0.6,
                 payout_ratio: float = 1,
                 starting_money: float = 25,
                 max_bets: int = 300,
                 max_money: int = 250):
        self.WIN_PROBABILITY = win_probability
        self.PAYOUT_RATIO = payout_ratio
        self.STARTING_MONEY = starting_money
        self.MAX_NUM_BETS = max_bets
        self.MAX_MONEY = max_money
        self.reset()

    def get_state(self) -> float:
        return np.round(self._player_money)

    def reset(self) -> float:
        self._player_money = self.STARTING_MONEY
        self._bets_remaining = self.MAX_NUM_BETS
        self._finished = False
        return self.get_state()

    def step(self, bet_size: float):

        if self._finished:
            raise AttributeError("Game has finished")
        if bet_size > self._player_money:
            raise ValueError(f"Cannot bet ${bet_size} when player has ${self._player_money}")

        player_money_before_bet = self._player_money

        # place the bet
        self._player_money -= bet_size

        # flip the coin
        won_bet = np.random.random() < self.WIN_PROBABILITY
        if won_bet:
            self._player_money += bet_size  # return the bet
            self._player_money += bet_size * self.PAYOUT_RATIO  # also give the payout for winning

        self._bets_remaining -= 1

        if not 0 <= self._player_money <= self.MAX_MONEY:
            self._player_money = np.clip(self._player_money, a_min=0, a_max=self.MAX_MONEY)
            self._finished = True

        if self._bets_remaining <= 0:
            self._finished = True

        debug_info = {
            "bet_size": bet_size,
            "player_money_before_bet": player_money_before_bet,
            "won_bet": won_bet,
            "player_money_after_bet": self._player_money
        }

        return self._player_money, self._player_money - player_money_before_bet, self._finished, debug_info


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
            money, reward, finished, debug_info = env.step(bet_size=np.round(money * 0.2))

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

    def __init__(self, Q_values=None):
        if Q_values is not None:
            self._Q = Q_values
        else:
            # todo: this makes assumptions about env
            self._Q = np.full(shape=(251, 250), fill_value=-25)

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


def test_agent(train_from_fresh, num_episodes=100):
    env = Environment()

    if train_from_fresh:
        agent = Agent()
        total_rewards = agent.train(env, start_epsilon=1, episodes=num_episodes)

        # plot running mean of rewards during training
        total_rewards_running_mean = np.convolve(total_rewards, np.ones(20) / 20, mode='valid')
        plt.plot(total_rewards_running_mean)
        plt.plot(np.linspace(0.8 * 250, 0, num_episodes))
        plt.title("Rewards during training")
        plt.show()

    else:
        agent = Agent(Q_values=np.loadtxt("Q.csv"))

    # plot Q values
    np.savetxt("Q.csv", agent._Q, fmt='%.2f')
    plt.imshow(agent._Q)
    plt.colorbar()
    plt.title("Q values heatmap")
    plt.show()

    # plot policy
    policy = [agent.get_action(s, explore_probability=0) for s in range(1, 250)]
    plt.plot(policy, label="Learned greedy policy")
    plt.title("Learned greedy policy (pure exploit mode, ie. epsilon=0)")
    optimal_policy = [min(250 - s, 0.2 * s) for s in range(1, 250)]
    plt.plot(optimal_policy, label="Optimal behaviour (Kelly Criterion)")
    plt.legend()
    plt.show()


test_agent(train_from_fresh=False)
