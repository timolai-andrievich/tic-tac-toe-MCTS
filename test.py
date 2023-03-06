"""Contains various unit tests.
"""
import os
import os.path

import numpy as np

from tic_tac_toe_mcts.game import Game, test_game, test_position, START_POSITION  # pylint: disable=unused-import
from tic_tac_toe_mcts import mcts
from tic_tac_toe_mcts import policy
from tic_tac_toe_mcts.config import Config

unit_test_config = Config()
unit_test_config.mcts_playout = 20
unit_test_config.test_games = 1


def test_model():
    """Runs unit tests on a model.Model class.
    """
    config = unit_test_config
    model = policy.Model(config)
    pos = START_POSITION.copy()
    pos.get_state()
    model.policy_function(pos.get_state())
    batch = (
        pos.get_state().reshape(
            (-1, Game.board_height, Game.board_width, Game.num_layers)),
        np.ones((1, Game.num_actions)) / Game.num_actions,
        np.array([[0, 1, 0]]),
    )
    model.train(config, batch)
    act, val = model.policy_function(pos.get_state())
    if not os.path.exists('./models'):
        os.mkdir('./models')
    model.save(file_name="./models/test")
    model = policy.Model(config, file_path="./models/test")
    new_act, new_val = model.policy_function(pos.get_state())
    assert (act - new_act).sum() < 1e-3
    assert (val - new_val).sum() < 1e-3


def test_node():
    """Runs unit tests on a mcts.Node class.
    """
    root = mcts.Node(None, 0, unit_test_config)
    game = Game()
    probs = np.ones(Game.num_actions) / Game.num_actions
    assert root.is_leaf()
    assert root.is_root()
    root.expand(game, probs)
    _, node = root.select()
    node.update_recursive(np.array([0, 1, 0]))
    assert (root.results == np.array([0, 1, 0])).all()
    assert (node.results == np.array([0, 1, 0])).all()


def test_tree():
    """Runs unit tests on a mcts.MCTS class.
    """
    # Create objects required for running tests
    model = policy.Model(unit_test_config)
    game = Game()
    tree = mcts.MCTS(unit_test_config)
    # Run MCTS on a blank game and freshly initialized NN
    tree.run(game, model.policy_function)


if __name__ == '__main__':
    print('Run "pytest test.py" in the project directory.')
