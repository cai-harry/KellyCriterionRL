import sys

from src import Environment


class UserAgent:
    """
    An console agent that asks the user to type in bet sizes.
    """

    def act(self, env: Environment):
        env.render(mode="human")
        bet_size = _get_action_from_user_input(env)
        return bet_size


def _get_action_from_user_input(env: Environment):
    try:
        input_integer = int(input(">> Enter bet size: $"))
    except ValueError:
        print("Value must be an integer", file=sys.stderr)
        return _get_action_from_user_input(env)
    if input_integer in env.get_legal_actions():
        return input_integer
    else:
        print("Cannot bet that much", file=sys.stderr)
        return _get_action_from_user_input(env)


if __name__ == "__main__":
    env = Environment()
    user_agent = UserAgent()
    finished = False
    while not finished:
        action = user_agent.act(env)
        print("\n\nFlipping coin...")
        state, reward, finished, debug_info = env.step(action)
        if reward > 0:
            print("WON :)\n")
        else:
            print("LOST :(\n")
