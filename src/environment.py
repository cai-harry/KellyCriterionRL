import numpy as np


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
