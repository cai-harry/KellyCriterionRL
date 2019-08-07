import pathlib
import tempfile

import numpy as np

from src.context import MODELS_DIR
from src.environment import Environment
from src.agent import Agent

# Determines how long the model training portion of test_train() takes
DESIRED_TRAIN_NUM_SECONDS = 1
APPROX_EPISODES_PER_SECOND = 8000


def test_train():
    train_num_episodes = APPROX_EPISODES_PER_SECOND * DESIRED_TRAIN_NUM_SECONDS

    # run a quick session of training and make plots.
    env = Environment()
    agent = Agent(env)
    agent.train(env,
                num_episodes=train_num_episodes,
                plot_training_rewards=False)

    # assert the trained agent has different Q values to a freshly instantiated one.
    fresh_env = Environment()
    fresh_agent = Agent(fresh_env)
    assert not np.array_equal(
        fresh_agent._Q,
        agent._Q
    )


def test_load_parameters():
    env = Environment()

    # instantiate an agent, loading in Q values from models/test
    agent = Agent(env, load_from_directory=MODELS_DIR / "test", load_N=False)
    # models/test/Q.csv is all 0's apart from a single 1 at [2, 1]
    assert agent._Q[2, 1] == 1.0

    # check that loading works in a two-step process too
    # also check that N values are correctly loaded
    agent = Agent(env)
    agent.load_parameters(MODELS_DIR / "test", load_N=True)
    assert agent._Q[2, 1] == 1.0
    assert agent._N[2, 1] == 1  # models/test/N.csv follows same pattern


def test_save_parameters():
    # create a temporary directory to save into. It gets deleted at the end of the test.
    with tempfile.TemporaryDirectory() as tmp:
        save_dir = pathlib.Path(tmp)
        env = Environment()

        agent = Agent(env)
        agent._Q[3, 1] = 1.0
        agent._N[3, 1] = 1
        agent.save_parameters(to_directory=save_dir)

        del agent
        new_agent = Agent(env, load_from_directory=save_dir)
        assert new_agent._Q[3, 1] == 1.0
        assert new_agent._N[3, 1] == 1
