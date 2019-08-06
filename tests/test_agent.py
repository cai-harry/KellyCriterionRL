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
    epsilon_fn = _epsilon_picket_fence(num_episodes=train_num_episodes,
                                       explore_ratio=0.9,
                                       repeats=5)

    # run a quick session of training and make plots.
    trained_agent = _run_training(train_from_fresh=True,
                                  num_episodes=train_num_episodes,
                                  epsilons=epsilon_fn,
                                  exploring_start=True)

    # assert the trained agent has different Q values to a freshly instantiated one.
    fresh_agent = Agent(Environment())
    assert not np.array_equal(
        fresh_agent._Q,
        trained_agent._Q
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


def _run_training(train_from_fresh, num_episodes, epsilons, exploring_start=False):
    env = Environment()

    if train_from_fresh:
        agent = Agent(env)

        total_rewards = agent.train(env,
                                    num_episodes=num_episodes, epsilons_each_episode=epsilons,
                                    exploring_start=exploring_start, plot_training_rewards=True)

        agent.save_parameters()

        agent.plot_N_values()

    else:
        agent = Agent(env, load_from_directory=MODELS_DIR / "latest")

    agent.plot_Q_values()

    agent.plot_policy(
        # also plot what the agent's policy would look like it it followed Kelly Criterion exactly
        optimal_policy=[min(250 - s, 0.2 * s) for s in range(1, 250)]
    )

    return agent


def _epsilon_picket_fence(num_episodes: int, explore_ratio: float, repeats: int):
    repeating_phase_length = num_episodes // repeats
    single_explore_phase_length = int(repeating_phase_length * explore_ratio)
    single_exploit_phase_length = repeating_phase_length - single_explore_phase_length

    single_repeating_phase = np.concatenate([
        np.ones(single_explore_phase_length),
        np.zeros(single_exploit_phase_length)
    ])

    return np.tile(single_repeating_phase, repeats)
