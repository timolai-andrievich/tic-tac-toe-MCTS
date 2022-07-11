from Game import Game, START_POSITION, test_game, test_position
from NN import NN, create_model
from MCTS import Node, MCST
import numpy as np
from typing import Tuple

def test_nnmodel():
    pos = START_POSITION.copy()
    state = pos.vectorize()[np.newaxis, ...]
    model = create_model()
    act, val = model(state)
    assert act.shape == (1, Game.num_actions)
    assert val.shape == (1, 3)


def test_nn():
    nn = NN()
    pos = START_POSITION.copy()
    state = pos.vectorize()
    nn.policy_function(pos)
    batch = [
        (pos.to_image(), (np.zeros(Game.num_actions), np.array([0, 1, 0]),),),
        (pos.to_image(), (np.zeros(Game.num_actions), np.array([0, 1, 0]),),),
    ]
    nn.train(batch)
    act, val = nn.policy_function(pos)
    nn.dump(file_name='../models/test')
    nn = NN(file_path='../models/test')
    nact, nval = nn.policy_function(pos)
    assert (act - nact).sum() < 1e-3
    assert (val - nval).sum() < 1e-3


def test_node():
    root = Node(None, 0, 1)
    game = Game()
    probs = np.ones(Game.num_actions) / Game.num_actions
    assert root.is_leaf()
    assert root.is_root()
    root.expand(game, probs)
    _: Tuple[int, Node] = root.select()
    action: int = _[0]
    node: Node = _[1]
    node.update_recursive(np.array([0, 1, 0]))
    assert (root.results == np.array([0, 1, 0])).all()
    assert (node.results == np.array([0, 1, 0])).all()


def test_tree():
    nn = NN()
    game = Game()
    tree = MCST()
    tree.run(game, nn.policy_function, 100)

if __name__ == "__main__":
    test_nn()
