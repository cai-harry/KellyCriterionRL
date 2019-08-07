import gym
from gym.utils import seeding
import numpy as np

GYM_ENV_ID = "coinflips-v0"

gym.envs.registration.register(
    id=GYM_ENV_ID,
    entry_point="src.environment:Environment"
)


class Environment(gym.Env):
    """
    The coinflips environment. Should implement the OpenAI Gym API.
    """

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

        # these attributes are required to implement the gym.Env API
        num_actions = self._MAX_MONEY  # 0 to max exclusive; can't bet exactly max
        num_states = self._MAX_MONEY + 1  # 0 to max inclusive
        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = gym.spaces.Discrete(num_states)

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

        # gym.Env API expects the returned reward value to be a float
        reward = float(self._player_money - player_money_before_bet)

        return self.get_state(), reward, self._finished, debug_info

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

    def render(self, mode='human'):
        state_str = f"Game finished: {self._finished}\n" \
                    f"Player money: ${self._player_money}" \
                    f"Bets taken: {self._MAX_NUM_BETS - self._bets_remaining}/{self._MAX_NUM_BETS}"
        if mode == "human":
            print(state_str)
        elif mode == "ansi":
            return state_str
        else:
            super(Environment, self).render(mode=mode)

    def seed(self, seed=None):
        # copied and pasted from gym mountain-car example
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self) -> float:
        return np.round(self._player_money, decimals=2)
