# %% Setup
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

plt.style.use("ggplot")


# %% Environment

class Environment:
    NUM_STATES = None
    NUM_ACTIONS = None

    _MAX_MONEY = None
    _WIN_PROBABILITY = None
    _PAYOUT_RATIO = None
    _STARTING_MONEY = None
    _MAX_NUM_BETS = None

    _player_money = None
    _bets_remaining = None
    _finished = None

    def __init__(self,
                 win_probability: float = 0.6,
                 payout_ratio: float = 1,
                 starting_money: float = 25,
                 max_bets: int = 300,
                 max_money: int = 250):
        self._WIN_PROBABILITY = win_probability
        self._PAYOUT_RATIO = payout_ratio
        self._STARTING_MONEY = starting_money
        self._MAX_NUM_BETS = max_bets
        self._MAX_MONEY = max_money
        self.reset()

        self.NUM_STATES = self._MAX_MONEY + 1  # 0 to max inclusive
        self.NUM_ACTIONS = self._MAX_MONEY  # 0 to max exclusive; can't bet exactly max

    def get_state(self) -> float:
        return int(np.round(self._player_money))

    def reset(self, random_starting_money: bool = False) -> float:
        if random_starting_money:
            # draw from triangular distribution from 1 to N such that each m is m-times more likely to be chosen than 1
            # this is because each m has m actions associated - we are trying to sample the Q-space evenly
            random_trianglular = np.random.triangular(
                left=1,
                mode=self._MAX_MONEY,
                right=self._MAX_MONEY
            )
            self._player_money = int(np.floor(random_trianglular))
        else:
            self._player_money = self._STARTING_MONEY
        self._bets_remaining = self._MAX_NUM_BETS
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
        won_bet = np.random.random() < self._WIN_PROBABILITY
        if won_bet:
            self._player_money += bet_size  # return the bet
            self._player_money += bet_size * self._PAYOUT_RATIO  # also give the payout for winning

        self._bets_remaining -= 1

        if not 0 < self._player_money < self._MAX_MONEY:
            self._player_money = np.clip(self._player_money, a_min=0, a_max=self._MAX_MONEY)
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

        np.savetxt("Q.csv", self._Q, fmt='%.2f')
        np.savetxt("N.csv", self._N, fmt='%d')

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


def test_agent(train_from_fresh, num_episodes, epsilons, exploring_start=False):
    env = Environment()

    if train_from_fresh:
        agent = Agent(env)

        total_rewards = agent.train(env, episodes=num_episodes, epsilons_each_episode=epsilons,
                                    exploring_start=exploring_start)

        # plot running mean of rewards during training
        running_mean_N = num_episodes // 100
        total_rewards_running_mean = np.convolve(total_rewards, np.ones(running_mean_N) / running_mean_N, mode='valid')
        plt.plot(total_rewards_running_mean, label="Total rewards (running mean with N=100)")

        # plot epsilon decay curve
        plt.plot(epsilons * np.max(total_rewards_running_mean), label="Epsilon decay curve (not to scale)")
        plt.title("Rewards during training")
        plt.xlabel("Episode")
        plt.ylabel("Reward ($)")
        plt.legend()
        plt.show()

        agent.plot_N_values()

    else:
        agent = Agent(Q_values=np.loadtxt("Q.csv"))

    agent.plot_Q_values()

    agent.plot_policy(
        optimal_policy=[min(250 - s, 0.2 * s) for s in range(1, 250)]
    )


# %% Define picket fence epsilon function
def epsilon_picket_fence(num_episodes: int, explore_ratio: float, repeats: int):
    repeating_phase_length = num_episodes // repeats
    single_explore_phase_length = int(repeating_phase_length * explore_ratio)
    single_exploit_phase_length = repeating_phase_length - single_explore_phase_length

    single_repeating_phase = np.concatenate([
        np.ones(single_explore_phase_length),
        np.zeros(single_exploit_phase_length)
    ])

    return np.tile(single_repeating_phase, repeats)


# %% Run training

DESIRED_TRAIN_NUM_SECONDS = 5 * 60
APPROX_EPISODES_PER_SECOND = 8000
train_num_episodes = APPROX_EPISODES_PER_SECOND * DESIRED_TRAIN_NUM_SECONDS

EPSILON_FN = epsilon_picket_fence(num_episodes=train_num_episodes,
                                  explore_ratio=0.9,
                                  repeats=5)

test_agent(train_from_fresh=True,
           num_episodes=train_num_episodes,
           epsilons=EPSILON_FN,
           exploring_start=True)
