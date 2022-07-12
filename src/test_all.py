import numpy as np

from Game import Game, START_POSITION, test_position, test_game
from MCTS import Node, MCTS
from NN import NN, create_model
from config import Config


unit_test_config = Config()
unit_test_config.mcts_playout = 20
unit_test_config.test_games = 1


def test_nnmodel():
    pos = START_POSITION.copy()
    state = pos.vectorize()[np.newaxis, ...]
    model = create_model()
    act, val = model(state)
    assert act.shape == (1, Game.num_actions)
    assert val.shape == (1, 3)


def test_nn():
    config = unit_test_config
    nn = NN(config)
    pos = START_POSITION.copy()
    pos.vectorize()
    nn.policy_function(pos)
    batch = [
        (pos.to_image(), (np.zeros(Game.num_actions), np.array([0, 1, 0]),),),
        (pos.to_image(), (np.zeros(Game.num_actions), np.array([0, 1, 0]),),),
    ]
    nn.train(config, batch)
    act, val = nn.policy_function(pos)
    nn.dump(file_name="../models/test")
    nn = NN(config, file_path="../models/test")
    new_act, new_val = nn.policy_function(pos)
    assert (act - new_act).sum() < 1e-3
    assert (val - new_val).sum() < 1e-3


def test_node():
    config = unit_test_config
    root = Node(None, 0, 1, config)
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
    nn = NN(unit_test_config)
    game = Game()
    tree = MCTS(unit_test_config)
    tree.run(game, nn.policy_function)
