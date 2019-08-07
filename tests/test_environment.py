import gym

from src.environment import Environment, GYM_ENV_ID


def test_gym_env_compatibility():
    # Is this environment instantiable by calling gym.make(GYM_ENV_ID)?
    env = gym.make(GYM_ENV_ID)
    assert env is not None

    # Have the main API methods been correctly implemented?
    # ie. can a dummy agent complete an episode in this environment without raising an exception?
    _run_dummy_agent_episode(env)


def test_get_state():
    expected_state = 22
    env = Environment(starting_money=expected_state)
    actual_state = env.get_state()
    assert expected_state == actual_state, \
        f"Expected state={expected_state}, got state={actual_state}"


def test_reset():
    env = Environment(starting_money=20, max_bets=200)
    env.step(bet_size=10)
    env.reset()
    assert env.get_state() == 20
    assert env._bets_remaining == 200


def test_step():
    test_cases = [
        {
            "summary": "1. Player bets $10 and wins",
            "env_args": {
                "win_probability": 1
            },
            "bet_size": 10,
            "expected_state": 35,
            "expected_reward": 10,
            "expected_finished": False,
        },
        {
            "summary": "2. Player bets $10 and loses",
            "env_args": {
                "win_probability": 0
            },
            "bet_size": 10,
            "expected_state": 15,
            "expected_reward": -10,
            "expected_finished": False,
        },
        {
            "summary": "3. Player goes bankrupt",
            "env_args": {
                "win_probability": 0
            },
            "bet_size": 25,
            "expected_state": 0,
            "expected_reward": -25,
            "expected_finished": True,
        },
        {
            "summary": "4. Player reaches max_money and wins",
            "env_args": {
                "win_probability": 1,
                "max_money": 30,
            },
            "bet_size": 5,
            "expected_state": 30,
            "expected_reward": 5,
            "expected_finished": True,
        },
        {
            "summary": "5. Player reaches max_bets and game ends",
            "env_args": {
                "max_bets": 1
            },
            "bet_size": 0,
            "expected_state": 25,
            "expected_reward": 0,
            "expected_finished": True,
        },
    ]

    for case in test_cases:
        env = Environment(**case['env_args'])
        state, reward, finished, debug_info = env.step(bet_size=case['bet_size'])
        assert state == case['expected_state'], \
            f"Expected state={case['expected_state']}, got state={state}"
        assert reward == case['expected_reward'], \
            f"Expected reward={case['expected_reward']}, got reward={reward}"
        assert finished == case['expected_finished']


def _run_dummy_agent_episode(env):
    """Inspiration: https://github.com/openai/gym/blob/master/examples/agents/random_agent.py"""
    agent = DummyAgent()
    episode_count = 2
    reward = 0
    done = False
    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, debug_info = env.step(action)
            if done:
                break
    env.close()


class DummyAgent(object):
    """
    A dummy agent designed to work with a gym.Env
    """
    def act(self, observation, reward, done):
        return 0
