import numpy as np

from src.environment import Environment


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
